#!/usr/bin/env python3
"""
Process mead3d dataset - creates symlinks to original data.
Run this script from the modelmead directory before training.
"""

import os
import sys

# Data paths
TEMPLATE_PKL_SRC = ""
TEMPLATE_PLY_SRC = ""
VERTICES_SRC = ""
WAV_SRC = ""

def create_symlinks():
    """Create symlinks to original mead3d data."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create directories
    os.makedirs(os.path.join(script_dir, "templates"), exist_ok=True)

    # Create symlinks
    links = [
        (TEMPLATE_PKL_SRC, os.path.join(script_dir, "templates.pkl")),
        (TEMPLATE_PLY_SRC, os.path.join(script_dir, "templates", "flame_sample.ply")),
        (VERTICES_SRC, os.path.join(script_dir, "vertices_npy")),
        (WAV_SRC, os.path.join(script_dir, "wav")),
    ]

    for src, dst in links:
        if os.path.exists(dst):
            if os.path.islink(dst):
                print(f"Skipping existing symlink: {dst}")
            else:
                print(f"Warning: {dst} exists but is not a symlink")
        elif os.path.exists(src):
            os.symlink(src, dst)
            print(f"Created symlink: {dst} -> {src}")
        else:
            print(f"Error: Source not found: {src}")
            sys.exit(1)

    print("\nData preparation complete!")
    print("\nThen flatten motion data:")
    print("  python preprocess_mead3d_flatten.py --dataset . --vertices_path vertices_npy \\")
    print("    --output_vertices_path vertices_npy_flat")
    print("\nTo train on mead3d:")
    print("  python main.py --dataset . --vertice_dim 15069 --feature_dim 64 --period 30 \\")
    print("    --train_subjects \"M003 M005 ...\" \\")
    print("    --val_subjects \"M032 ...\" \\")
    print("    --test_subjects \"M037 ...\" \\")
    print("    --wav_path wav --vertices_path vertices_npy_flat --template_file templates.pkl")

if __name__ == "__main__":
    create_symlinks()
