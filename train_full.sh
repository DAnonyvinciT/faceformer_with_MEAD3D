#!/usr/bin/env bash
set -euo pipefail

# 全量数据训练脚本（modelmead）
# 用法：
#   bash train_full.sh
#   bash train_full.sh --max_epoch 50 --save_path save_full_v2
# 说明：
#   1) 默认使用当前主训练配置（teacher_forcing=false, use_audio_cache=true）。
#   2) 数据由 data_loader.py 按 neutral-only（emo=0, level=0）规则筛选。
#   3) 其余参数可通过命令行追加覆盖。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/tkface/miniconda3/envs/faceformer/bin/python"

# 是否跳过离线音频缓存预处理：0 表示执行，1 表示跳过
SKIP_AUDIO_CACHE_PRECOMPUTE="${SKIP_AUDIO_CACHE_PRECOMPUTE:-0}"
SKIP_AUDIO_CACHE_PRECOMPUTE=1
if [[ "$SKIP_AUDIO_CACHE_PRECOMPUTE" != "1" ]]; then
    "$PYTHON_BIN" precompute_audio_cache.py \
        --dataset . \
        --wav_path wav \
        --audio_cache_dir audio_cache \
        --cache_dtype float32 \
        --overwrite false
fi

# 是否跳过顶点与模板展平预处理：0 表示执行，1 表示跳过
SKIP_MOTION_PREPROCESS="${SKIP_MOTION_PREPROCESS:-0}"
SKIP_MOTION_PREPROCESS=1
if [[ "$SKIP_MOTION_PREPROCESS" != "1" ]]; then
    "$PYTHON_BIN" preprocess_mead3d_flatten.py \
        --dataset . \
        --vertices_path vertices_npy \
        --output_vertices_path vertices_npy_flat \
        --overwrite false
fi

"$PYTHON_BIN" main.py \
    --dataset . \
    --vertice_dim 15069 \
    --feature_dim 128 \
    --period 25 \
    --train_subjects "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029" \
    --val_subjects "M032 M033 M034 M035 W033 W035 W036" \
    --test_subjects "M037 M039 M040 M041 M042 W037 W038 W040" \
    --wav_path wav \
    --vertices_path vertices_npy_flat \
    --gradient_accumulation_steps 1 \
    --max_epoch 30 \
    --template_file templates.pkl \
    --teacher_forcing false \
    --num_workers 4 \
    --pin_memory true \
    --persistent_workers true \
    --prefetch_factor 2 \
    --use_audio_cache true \
    --audio_cache_dir audio_cache \
    --save_path save_full \
    --result_path result_full \
    "$@"
