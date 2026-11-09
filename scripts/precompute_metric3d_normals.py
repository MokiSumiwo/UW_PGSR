import os
import argparse
import glob
from typing import List

import cv2
import torch
from tqdm import tqdm


def list_images(img_dir: str, exts: List[str] = None):
    if exts is None:
        exts = [".png", ".jpg", ".jpeg"]
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    img_paths = sorted(img_paths)
    return img_paths


def load_metric3d_model(device: str = "cuda", repo_dir: str = None):
    """
    Load Metric3D ViT-small model.

    默认从本地 metric3d 仓库加载，避免再从 GitHub 拉代码。
    仍然会通过 torch.hub.load_state_dict_from_url 从 HuggingFace 下载权重（首次需要外网）。
    """
    if repo_dir is None:
        raise ValueError("repo_dir for Metric3D is None. Please set --metric3d_repo to local metric3d path.")

    print(f"[Metric3D] Start loading model from local repo: {repo_dir}")
    try:
        model = torch.hub.load(
            repo_dir,
            "metric3d_vit_small",
            pretrain=True,
            source="local",
        )
    except Exception as e:
        print("[Metric3D] ERROR: local torch.hub.load failed:")
        print(f"           {type(e).__name__}: {e}")
        print("           请检查 repo_dir 路径是否正确，以及是否满足 Metric3D 的依赖环境。")
        raise
    print("[Metric3D] Model loaded successfully (local).")
    model = model.to(device).eval()
    return model


def preprocess_image_bgr_to_tensor(img_bgr, device: str):
    """
    img_bgr: H x W x 3, uint8 (OpenCV)
    return: 1 x 3 x H x W, float32 in [0,1]
    """
    # 注意：img_bgr[:, :, ::-1] 会产生 stride 为负的 numpy 视图，
    # 需要 .copy() 之后再转成 tensor，避免 PyTorch 报 stride 错误。
    img_rgb = img_bgr[:, :, ::-1].copy()  # BGR -> RGB，并拷贝成连续数组
    img_rgb = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    return img_rgb.unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Metric3D normals for all training images."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="COLMAP/Blender scene root, same as --source_path in train.py",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="images/ColorImage",
        help="Relative path from source_path to images folder (matches ModelParams._images).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="metric3d_normals",
        help="Relative folder under source_path to save .pt files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Metric3D inference (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--resize_short",
        type=int,
        default=0,
        help="Optionally resize images so that the short side = this value (0 = no resize).",
    )
    parser.add_argument(
        "--metric3d_repo",
        type=str,
        default="/mnt/new/home_old/ycy/metric3d",
        help="本地 Metric3D 仓库路径（包含 hubconf.py）.",
    )

    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    img_dir = os.path.join(source_path, args.images)
    out_root = os.path.join(source_path, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    print(f"[Metric3D] Source path: {source_path}")
    print(f"[Metric3D] Image dir : {img_dir}")
    print(f"[Metric3D] Output dir: {out_root}")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    img_paths = list_images(img_dir)
    print(f"[Metric3D] Found {len(img_paths)} images.")

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    print(f"[Metric3D] Using device: {device}")

    # 在列出图片之后再加载模型，方便确认程序卡在“加载模型”还是“处理图片”阶段
    model = load_metric3d_model(device=device, repo_dir=args.metric3d_repo)

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Pre-computing Metric3D normals"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_root, base + ".pt")
            if os.path.exists(out_path):
                # Skip if already computed
                continue

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[Metric3D] Warning: failed to read image {img_path}, skip.")
                continue

            # Optional resize to reduce GPU memory
            if args.resize_short > 0:
                h, w = img_bgr.shape[:2]
                short = min(h, w)
                if short != args.resize_short:
                    scale = args.resize_short / float(short)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb = preprocess_image_bgr_to_tensor(img_bgr, device=device)  # [1,3,H,W]

            try:
                pred_depth, confidence, output_dict = model.inference({"input": rgb})
            except Exception as e:
                print(f"[Metric3D] Error running inference on {img_path}: {e}")
                continue

            if "prediction_normal" not in output_dict:
                print(
                    f"[Metric3D] prediction_normal not in output_dict for {img_path}, "
                    f"please ensure you are using Metric3Dv2 (ViT model)."
                )
                continue

            # prediction_normal: [B,4,H,W] -> [1,3,H,W] & [1,1,H,W]
            pred_normal_all = output_dict["prediction_normal"]  # [1,4,H,W]
            metric_normal = pred_normal_all[:, :3, :, :].contiguous().cpu()
            normal_conf = pred_normal_all[:, 3:4, :, :].contiguous().cpu()

            torch.save(
                {
                    "metric_normal": metric_normal,        # [1,3,H,W], float32
                    "metric_normal_conf": normal_conf,    # [1,1,H,W], float32
                },
                out_path,
            )

    print("[Metric3D] Done. All available normals saved.")


if __name__ == "__main__":
    main()


