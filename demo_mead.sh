#!/usr/bin/env bash
set -euo pipefail

# modelmead 推理脚本（固定参数）
python demo.py \
  --dataset . \
  --model_name save_full/30_model \
  --vertice_dim 15069 \
  --feature_dim 128 \
  --period 25 \
  --train_subjects "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029" \
  --test_subjects "M037 M039 M040 M041 M042 W037 W038 W040" \
  --wav_path ../demo/wav/test.wav \
  --result_path demo/result_from_30 \
  --output_path demo/output_from_30 \
  --condition M003 \
  --subject M003
