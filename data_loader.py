import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa

class Dataset(data.Dataset):
    """支持流式加载的数据集。"""
    def __init__(self, data, subjects_dict, data_type="train", dataset_name="vocaset"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.use_audio_cache = False
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        # 延迟初始化 processor，降低多进程加载时的瞬时内存压力
        self._processor = None
        self._cache_miss_warned = False

    def set_audio_cache(self, use_audio_cache):
        """设置是否优先读取离线音频缓存。"""
        self.use_audio_cache = use_audio_cache

    @property
    def processor(self):
        if self._processor is None:
            self._processor = Wav2Vec2Processor.from_pretrained("/home/tkface/workspace/FaceFormer/wav2vec2-base-960h")
        return self._processor

    def __getitem__(self, index):
        """按需读取单条样本（音频 + 顶点 + 模板）。"""
        item = self.data[index]
        file_name = item["name"]

        # 优先读取离线缓存，缺失时回退到在线解码
        audio = None
        audio_cache_path = item.get("audio_cache_path")
        if self.use_audio_cache and audio_cache_path and os.path.exists(audio_cache_path):
            audio = np.load(audio_cache_path, allow_pickle=False)
        else:
            if self.use_audio_cache and (not self._cache_miss_warned):
                print("[WARN] 音频缓存缺失，已回退到在线解码。")
                self._cache_miss_warned = True
            speech_array, _ = librosa.load(item["audio_path"], sr=16000)
            audio = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)

        vertice = np.load(item["vertice_path"], allow_pickle=True)

        template = item["template"]

        if self.data_type == "train":
            # mead3d format: M003_002_0_0 -> M003; vocaset format: FaceTalk_xxx_Sentence01 -> FaceTalk_xxx
            subject = self._extract_subject(file_name)
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels

        return (torch.FloatTensor(audio),
            torch.FloatTensor(vertice),
            torch.FloatTensor(template),
                torch.FloatTensor(one_hot),
                file_name)

    def _extract_subject(self, filename):
        """从文件名中提取说话人 ID。"""
        # mead3d: M003_002_0_0.npy -> M003 (first field)
        # vocaset: FaceTalk_xxx_Sentence01.npy -> FaceTalk_xxx (all but last field)
        if self.dataset_name in ["modelmead", "."]:
            return filename.split("_")[0]
        return "_".join(filename.split("_")[:-1])

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data (streaming mode)...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    # When dataset is ".", use current directory; otherwise use relative path from cwd
    dataset_base = os.getcwd() if args.dataset == "." else os.path.join(os.getcwd(), args.dataset)

    audio_path = os.path.join(dataset_base, args.wav_path)
    vertices_path = os.path.join(dataset_base, args.vertices_path)

    # 先构建说话人划分字典
    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    all_subjects = set(subjects_dict["train"] + subjects_dict["val"] + subjects_dict["test"])

    template_file = os.path.join(dataset_base, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # 先按说话人名单做一级过滤
    splits = {'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
              'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)}}
    dataset_key = args.dataset if args.dataset != "." else "."

    modelmead_sentence_rank = {}
    modelmead_train_end_rank = {}
    if args.dataset in ["modelmead", "."]:
        subject_to_items = defaultdict(list)
        for r, ds, fs in os.walk(audio_path):
            for f in fs:
                if not f.endswith("wav"):
                    continue
                key = f.replace("wav", "npy")
                subject_id = key.split("_")[0]
                if subject_id not in all_subjects:
                    continue

                parts = os.path.splitext(f)[0].split("_")
                if len(parts) < 4:
                    continue
                try:
                    seq_id = int(parts[1])
                    emo_id = int(parts[2])
                    level_id = int(parts[3])
                except ValueError:
                    continue

                # 只保留 neutral 样本：emo=0 且 level=0
                if emo_id != 0 or level_id != 0:
                    continue

                subject_to_items[subject_id].append((seq_id, f))

        for subject_id, items in subject_to_items.items():
            # 在 neutral 子集内，按句子序号稳定排序并重编号 sentence_id（从 1 开始）
            items.sort(key=lambda x: (x[0], x[1]))
            for idx, (_, file_name) in enumerate(items, start=1):
                modelmead_sentence_rank[(subject_id, file_name)] = idx
            # 按说话人做 80/20 动态切分：前 80% 训练，后 20% 验证/测试
            train_end = int(math.floor(len(items) * 0.8))
            train_end = max(1, min(train_end, len(items)))
            modelmead_train_end_rank[subject_id] = train_end

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                key = f.replace("wav", "npy")
                # mead3d: M003_002_0_0 -> M003；vocaset: FaceTalk_xxx_Sentence01 -> FaceTalk_xxx
                if args.dataset in ["modelmead", "."]:
                    subject_id = key.split("_")[0]
                    parts = os.path.splitext(f)[0].split("_")
                    if len(parts) < 4:
                        continue
                    try:
                        emo_id = int(parts[2])
                        level_id = int(parts[3])
                    except ValueError:
                        continue
                    if emo_id != 0 or level_id != 0:
                        continue

                    sentence_id = modelmead_sentence_rank.get((subject_id, f))
                    if sentence_id is None:
                        continue
                else:
                    subject_id = "_".join(key.split("_")[:-1])
                    sentence_id = int(key.split(".")[0][-2:])

                # 不在 train/val/test 说话人名单内则跳过
                if subject_id not in all_subjects:
                    continue

                # 根据 sentence_id 判断当前样本归属 split
                if args.dataset in ["modelmead", "."]:
                    train_end = modelmead_train_end_rank.get(subject_id)
                    if train_end is None:
                        continue
                    in_train = subject_id in subjects_dict["train"] and sentence_id <= train_end
                    in_val = subject_id in subjects_dict["val"] and sentence_id > train_end
                    in_test = subject_id in subjects_dict["test"] and sentence_id > train_end
                else:
                    in_train = subject_id in subjects_dict["train"] and sentence_id in splits[dataset_key]['train']
                    in_val = subject_id in subjects_dict["val"] and sentence_id in splits[dataset_key]['val']
                    in_test = subject_id in subjects_dict["test"] and sentence_id in splits[dataset_key]['test']

                if not (in_train or in_val or in_test):
                    continue

                wav_path = os.path.join(r, f)
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))

                # 顶点文件缺失时跳过
                if not os.path.exists(vertice_path):
                    continue

                # 计算离线音频缓存路径
                rel_wav_path = os.path.relpath(wav_path, audio_path)
                cache_rel_path = os.path.splitext(rel_wav_path)[0] + ".npy"
                audio_cache_path = os.path.join(dataset_base, args.audio_cache_dir, cache_rel_path)

                # 仅当模板存在时写入样本
                if subject_id in templates:
                    temp = templates[subject_id]
                    data[key]["name"] = f
                    data[key]["template"] = temp.reshape((-1))
                    data[key]["audio_path"] = wav_path
                    data[key]["vertice_path"] = vertice_path
                    data[key]["audio_cache_path"] = audio_cache_path

                    if in_train:
                        train_data.append(data[key])
                    if in_val:
                        valid_data.append(data[key])
                    if in_test:
                        test_data.append(data[key])

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    dataset_name = args.dataset if args.dataset != "." else "modelmead"

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_dataset = Dataset(train_data, subjects_dict, "train", dataset_name)
    train_dataset.set_audio_cache(args.use_audio_cache)
    dataset["train"] = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, **loader_kwargs)

    valid_dataset = Dataset(valid_data, subjects_dict, "val", dataset_name)
    valid_dataset.set_audio_cache(args.use_audio_cache)
    dataset["valid"] = data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    test_dataset = Dataset(test_data, subjects_dict, "test", dataset_name)
    test_dataset.set_audio_cache(args.use_audio_cache)
    dataset["test"] = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    return dataset

if __name__ == "__main__":
    get_dataloaders()
    