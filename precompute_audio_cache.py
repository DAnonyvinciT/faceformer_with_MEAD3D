import argparse
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor


def parse_bool(value):
    """解析命令行布尔参数。"""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"不支持的布尔值: {value}")


def resolve_dataset_base(dataset_arg):
    """解析数据集根目录。"""
    if dataset_arg == ".":
        return os.getcwd()
    return os.path.join(os.getcwd(), dataset_arg)


def main():
    parser = argparse.ArgumentParser(description="离线预计算音频缓存")
    parser.add_argument("--dataset", type=str, default=".", help="数据集根目录，'.' 表示当前目录")
    parser.add_argument("--wav_path", type=str, default="wav", help="音频目录")
    parser.add_argument("--audio_cache_dir", type=str, default="audio_cache", help="缓存输出目录")
    parser.add_argument("--processor_path", type=str, default="wav2vec2-base-960h", help="Wav2Vec2Processor 路径")
    parser.add_argument("--cache_dtype", type=str, default="float32", choices=["float32", "float16"], help="缓存数据类型")
    parser.add_argument("--overwrite", type=parse_bool, default=False, help="是否覆盖已存在缓存")
    args = parser.parse_args()

    dataset_base = resolve_dataset_base(args.dataset)
    audio_root = os.path.join(dataset_base, args.wav_path)
    cache_root = os.path.join(dataset_base, args.audio_cache_dir)

    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"音频目录不存在: {audio_root}")

    os.makedirs(cache_root, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(args.processor_path)

    wav_files = []
    for root, _, files in os.walk(audio_root):
        for file_name in files:
            if file_name.endswith(".wav"):
                wav_files.append(os.path.join(root, file_name))

    wav_files.sort()
    if not wav_files:
        raise RuntimeError(f"未在目录中找到 wav 文件: {audio_root}")

    dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    processed = 0
    skipped = 0

    for wav_path in tqdm(wav_files, desc="PrecomputeAudioCache"):
        rel_path = os.path.relpath(wav_path, audio_root)
        cache_rel = os.path.splitext(rel_path)[0] + ".npy"
        cache_path = os.path.join(cache_root, cache_rel)
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_path) and (not args.overwrite):
            skipped += 1
            continue

        speech_array, _ = librosa.load(wav_path, sr=16000)
        audio = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio = np.asarray(audio, dtype=dtype)
        np.save(cache_path, audio)
        processed += 1

    print(
        f"缓存完成: processed={processed}, skipped={skipped}, total={len(wav_files)}, "
        f"cache_dir={Path(cache_root).resolve()}"
    )


if __name__ == "__main__":
    main()
