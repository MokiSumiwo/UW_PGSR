#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil


def run_cmd(cmd, cwd=None):
    print(f"[Run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="One-click: generate depths (Depth-Anything-V2) and write depth_params.json (3D-UIR) for UW_PGSR_4")
    parser.add_argument("-s", "--dataset", required=True, help="Dataset root (contains sparse/ and images/...)")
    parser.add_argument("--images_subdir", default="images/ColorImage", help="Relative images folder under dataset")
    parser.add_argument("-d", "--depths_subdir", default="depths", help="Relative depths output folder under dataset")

    parser.add_argument("--depth_anything_repo", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "3D-UIR", "Depth-Anything-V2"),
                        help="Path to Depth-Anything-V2 repo (default: ../3D-UIR/Depth-Anything-V2)")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"],
                        help="Depth-Anything-V2 encoder")
    parser.add_argument("--ckpt", default="", help="Optional path to depth_anything_v2_<encoder>.pth; if empty, will rely on repo checkpoints/")

    parser.add_argument("--uir_repo", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "3D-UIR"),
                        help="Path to 3D-UIR repo (default: ../3D-UIR)")
    parser.add_argument("--model_type", default="bin", choices=["bin", "txt"],
                        help="COLMAP model type for read (3D-UIR make_depth_scale.py)")

    args = parser.parse_args()

    dataset_abs = os.path.abspath(args.dataset)
    img_dir = os.path.join(dataset_abs, args.images_subdir)
    depths_dir = os.path.join(dataset_abs, args.depths_subdir)
    ensure_dir(depths_dir)

    # 1) Run Depth-Anything-V2 to generate inverse-depth PNGs
    run_py = os.path.join(os.path.abspath(args.depth_anything_repo), "run.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"Depth-Anything-V2 run.py not found at: {run_py}")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    # If checkpoint is provided, ensure it exists; then copy to repo checkpoints with standard name that run.py expects.
    # Many versions of Depth-Anything-V2 do not accept a --pretrained-model flag and auto-load from checkpoints/.
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Provided checkpoint not found: {args.ckpt}")
        repo_abs = os.path.abspath(args.depth_anything_repo)
        parent_dir = os.path.dirname(repo_abs)
        expected_name = f"depth_anything_v2_{args.encoder}.pth"
        # Case A: repo_root/checkpoints (some forks use this)
        repo_ckpt_dir = os.path.join(repo_abs, "checkpoints")
        os.makedirs(repo_ckpt_dir, exist_ok=True)
        expected_path_a = os.path.join(repo_ckpt_dir, expected_name)
        if os.path.abspath(args.ckpt) != os.path.abspath(expected_path_a):
            print(f"[Info] Copy checkpoint to {expected_path_a}")
            shutil.copyfile(args.ckpt, expected_path_a)
        # Case B: parent/Depth-Anything-V2/checkpoints (some run.py uses this relative path)
        nested_ckpt_dir = os.path.join(parent_dir, "Depth-Anything-V2", "checkpoints")
        os.makedirs(nested_ckpt_dir, exist_ok=True)
        expected_path_b = os.path.join(nested_ckpt_dir, expected_name)
        if not os.path.isfile(expected_path_b):
            try:
                print(f"[Info] Also copy checkpoint to {expected_path_b} (for relative path compatibility)")
                shutil.copyfile(args.ckpt, expected_path_b)
            except Exception:
                pass

    da_cmd = [
        sys.executable, run_py,
        "--encoder", args.encoder,
        "--pred-only",
        "--grayscale",
        "--img-path", img_dir,
        "--outdir", depths_dir
    ]

    # Run from parent directory so that 'Depth-Anything-V2/checkpoints/...' relative paths resolve
    repo_abs = os.path.abspath(args.depth_anything_repo)
    parent_dir = os.path.dirname(repo_abs)
    run_cmd(da_cmd, cwd=parent_dir)

    # 2) Run 3D-UIR's make_depth_scale.py to generate depth_params.json
    make_scale_py = os.path.join(os.path.abspath(args.uir_repo), "utils", "make_depth_scale.py")
    if not os.path.isfile(make_scale_py):
        raise FileNotFoundError(f"3D-UIR make_depth_scale.py not found at: {make_scale_py}")

    scale_cmd = [
        sys.executable, make_scale_py,
        "--base_dir", dataset_abs,
        "--depths_dir", depths_dir,
        "--model_type", args.model_type
    ]
    run_cmd(scale_cmd, cwd=os.path.dirname(make_scale_py))

    print("\nAll done.")
    print(f"- Depths saved to: {depths_dir}")
    print(f"- depth_params.json saved to: {os.path.join(dataset_abs, 'sparse', '0', 'depth_params.json')}")
    print("\nYou can now train UW_PGSR_4 with DRO:")
    print(f"python train.py -s {dataset_abs} -m <output_dir> -d {args.depths_subdir} --use_underwater_model --eval")


if __name__ == "__main__":
    main()


