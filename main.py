import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import logging
import sys
from datetime import datetime

import torch
import torch.nn as nn

from data_loader import get_dataloaders
from faceformer import Faceformer


class TqdmLoggingHandler(logging.Handler):
    """使用 tqdm.write 输出日志，避免打断进度条。"""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class StreamToLogger:
    """将 stdout/stderr 重定向到 logging，过滤空行与进度条控制符。"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_line(line)

    def flush(self):
        if self._buffer:
            self._log_line(self._buffer)
            self._buffer = ""

    def _log_line(self, line):
        line = line.rstrip("\r")
        stripped = line.strip()
        if not stripped:
            return
        # tqdm 进度条通常包含回车控制，避免写入日志
        if "\r" in line:
            return
        self.logger.log(self.level, stripped)


def get_progress_stream():
    """优先将 tqdm 输出到终端设备，避免被 stderr 重定向后写入日志。"""
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return sys.__stderr__


def sanitize_name(value):
    """将名称转换为安全文件名片段。"""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "unknown"


def build_log_file_path(args):
    """按规则生成日志文件名：train_模型名_数据集_年月日时分.log。"""
    dataset_base = os.path.dirname(os.path.abspath(__file__)) if args.dataset == "." else os.path.join(os.getcwd(), args.dataset)
    log_dir = os.path.join(dataset_base, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_name = sanitize_name(args.model_name)
    dataset_name = sanitize_name(os.path.basename(os.path.normpath(dataset_base)))
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    base_name = f"train_{model_name}_{dataset_name}_{timestamp}"
    log_file = os.path.join(log_dir, f"{base_name}.log")

    if not os.path.exists(log_file):
        return log_file

    suffix = 1
    while True:
        candidate = os.path.join(log_dir, f"{base_name}_{suffix:02d}.log")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def setup_logger(args):
    """配置训练日志：终端输出与文件输出同时开启。"""
    log_file = build_log_file_path(args)

    logger = logging.getLogger("faceformer_train")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.propagate = True

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(stream=sys.__stdout__)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if root_logger.handlers:
        root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 让常见第三方库透传到根日志器，避免绕过本地日志文件
    for lib_logger_name in ["transformers", "torch", "urllib3", "matplotlib"]:
        lib_logger = logging.getLogger(lib_logger_name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True

    return logger, log_file


def redirect_std_streams(logger):
    """将标准输出与错误输出统一收集到日志。"""
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


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


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, logger, epoch=100):
    dataset_base = os.path.dirname(os.path.abspath(__file__)) if args.dataset == "." else os.path.join(os.getcwd(), args.dataset)
    save_path = os.path.join(dataset_base, args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    progress_stream = get_progress_stream()
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            dynamic_ncols=True,
            file=progress_stream,
            mininterval=0.5,
        )
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template,  vertice, one_hot, criterion, teacher_forcing=args.teacher_forcing)
            loss.backward()

            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
        pbar.close()
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                cond_idx = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, cond_idx, :]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for cond_idx in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[cond_idx]
                    one_hot = one_hot_all[:, cond_idx, :]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                        
        current_loss = np.mean(valid_loss_log)
        train_loss = np.mean(loss_log)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        logger.info(
            "Epoch %d, iteration %d TRAIN LOSS:%.7f VALID LOSS:%.7f",
            e + 1,
            iteration,
            train_loss,
            current_loss,
        )
    if progress_stream not in {sys.__stdout__, sys.__stderr__}:
        progress_stream.close()
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    dataset_base = os.path.dirname(os.path.abspath(__file__)) if args.dataset == "." else os.path.join(os.getcwd(), args.dataset)
    result_path = os.path.join(dataset_base, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(dataset_base, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        if args.dataset in ["modelmead", "."]:
            train_subject = file_name[0].split("_")[0]
        else:
            train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            cond_idx = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, cond_idx, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for cond_idx in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[cond_idx]
                one_hot = one_hot_all[:, cond_idx, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy_flat", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--num_workers", type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument("--pin_memory", type=parse_bool, default=True, help='是否启用 DataLoader pin_memory')
    parser.add_argument("--persistent_workers", type=parse_bool, default=True, help='是否启用 DataLoader persistent_workers')
    parser.add_argument("--prefetch_factor", type=int, default=2, help='每个 worker 的预取批次数')
    parser.add_argument("--use_audio_cache", type=parse_bool, default=False, help='是否优先读取离线音频缓存')
    parser.add_argument("--audio_cache_dir", type=str, default="audio_cache", help='离线音频缓存目录（相对 dataset 路径）')
    parser.add_argument("--teacher_forcing", type=parse_bool, default=False, help='是否在训练时使用 teacher forcing')
    parser.add_argument("--model_name", type=str, default="faceformer", help='模型名称（用于日志文件命名）')
    parser.add_argument("--log_dir", type=str, default="logs", help='日志目录（相对 dataset 路径）')
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help='日志等级')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    logger, log_file = setup_logger(args)
    redirect_std_streams(logger)
    logger.info("训练启动，日志文件: %s", log_file)

    #build model
    model = Faceformer(args)
    logger.info("model parameters: %d", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, logger, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()