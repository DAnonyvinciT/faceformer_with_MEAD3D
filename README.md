# faceformer_with_MEAD3D

 FaceFormer 的衍生项目，核心目标是为 FaceFormer 增加 MEAD3D 数据集支持。并使用较新的python和pytorch版本，兼容50系以上显卡。

## 项目定位

相对原始 FaceFormer（主要面向 VOCASET 与 BIWI），modelmead 保留了原模型主体结构，同时补齐了在 MEAD3D 上可直接训练、推理和渲染的完整工程流程。与最初的 FaceFormer（主要面向 VOCASET 与 BIWI）相比，modelmead 保留了原模型的主体结构，同时完善了可在 MEAD3D 上直接进行训练、推理和渲染的完整工程流程。相较于最初的 FaceFormer（主要面向 VOCASET 与 BIWI），modelmead 保留了原模型的主体结构，同时完善了可在 MEAD3D 上直接进行训练、推理和渲染的完整工程流程。相较于最初的 FaceFormer（主要面向 VOCASET 与 BIWI），modelmead 保留了原模型的主体结构，同时完善了可在 MEAD3D 上直接进行训练、推理和渲染的完整工程流程。与最初的 FaceFormer（主要面向 VOCASET 与 BIWI）相比，modelmead 保留了原模型的主体结构，同时完善了可在 MEAD3D 上直接进行训练、推理和渲染的完整工程流程。

## 与 FaceFormer 的主要改动点

1. 数据集适配方式
- 支持使用 `--dataset .` 。
- 数据目录组织围绕 `wav`、`vertices_npy`、`templates.pkl` 与 `vertices_npy_flat` 展开。

2. 音频-顶点对齐逻辑
- 在 `faceformer.py` 中，`dataset=="."` 时采用与 BIWI 同类的 2:1 对齐策略（每帧顶点对应两个音频步）。
- 增加长度边界处理，避免音频特征长度与顶点帧长度不一致导致训练不稳定。

3. 数据加载与切分策略
- `data_loader.py` 采用流式读取，避免一次性加载全量样本造成内存压力。
- 增加 neutral-only 过滤（`emo=0` 且 `level=0`）。-增加仅中性过滤（`emo=0`且`level=0`）。
- 增加说话人内 80/20 动态切分规则（训练/验证测试）。
- 增加音频缓存优先读取机制：优先读取 `audio_cache`，缺失时回退到在线 `librosa + processor` 处理。

4. 新增数据预处理脚本
- `process_mead3d_data.py`：创建模板、顶点、音频软链接。
- `preprocess_mead3d_flatten.py`：将顶点由 `(T, V, 3)` 展平为 `(T, V*3)`。
- `precompute_audio_cache.py`：预计算音频输入缓存，降低训练时重复计算开销。

5. 训练与推理入口脚本
- `train_full.sh`：modelmead 的训练参数模板。
- `demo_mead.sh`：modelmead 的推理参数模板。

## 快速开始

0.环境
python = 3.9.25
pytorch = 2.8
cuda = 12.8


1. 数据准备
按照probtalk3d项目的数据集准备MEAD3D数据
https://github.com/uuembodiedsocialai/ProbTalk3D/tree/main

之后将数据目录写入process_mead3d_data.py脚本，创建软链接

```bash
python process_mead3d_data.py
```

2. 顶点展平预处理

```bash
python preprocess_mead3d_flatten.py --dataset . --vertices_path vertices_npy --output_vertices_path vertices_npy_flat
```

3. 音频缓存预计算（推荐）

```bash
python precompute_audio_cache.py --dataset . --wav_path wav --audio_cache_dir audio_cache
```

4. 训练

```bash
bash train_full.sh
```

5. 推理

```bash
bash demo_mead.sh
```

## 依赖说明

- 渲染链路额外依赖：
  - `ffmpeg`（系统命令）

## 致谢
感谢faceformer项目提供基本模型
感谢Probtalk3D提供MEAD3D数据集的获取与处理方式



