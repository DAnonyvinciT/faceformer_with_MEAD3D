
cd /home/tkface/workspace/FaceFormer/modelmead

python main.py \
    --dataset . \
    --vertice_dim 15069 \
    --feature_dim 128 \
    --period 30 \
    --train_subjects "M003" \
    --val_subjects "M032" \
    --test_subjects "M037" \
    --wav_path wav \
    --vertices_path vertices_npy_flat \
    --template_file templates.pkl \
    --max_epoch 5 \
    --save_path save_test \
    --result_path result_test
    "${EXTRA_ARGS[@]}"