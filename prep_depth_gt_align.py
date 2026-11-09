# prep_depth_gt_align.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import OpenEXR
import Imath


def read_exr_channel(file_path: str, channel_name: str = "R") -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    exr_file = OpenEXR.InputFile(file_path)
    try:
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())

        ch = channel_name if channel_name in channels else None
        if ch is None:
            lower_map = {c.lower(): c for c in channels}
            if channel_name.lower() in lower_map:
                ch = lower_map[channel_name.lower()]
            else:
                cand = [c for c in channels if c.split(".")[-1].lower() == channel_name.lower()]
                if len(cand) == 1:
                    ch = cand[0]
        if ch is None:
            raise KeyError(f"EXR 不含通道 {channel_name}。可用通道: {channels}")

        try:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float32)
        except Exception:
            pt = Imath.PixelType(Imath.PixelType.HALF)
            buf = exr_file.channel(ch, pt)
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)

        return arr.reshape((height, width))
    finally:
        del exr_file


def write_exr_single_channel(out_path: str, data: np.ndarray, channel_name: str = "R") -> None:
    """
    写单通道 float32 EXR。
    """
    data = np.asarray(data, dtype=np.float32)
    H, W = data.shape

    header = OpenEXR.Header(W, H)
    ch = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = {channel_name: ch}

    out = OpenEXR.OutputFile(out_path, header)
    try:
        out.writePixels({channel_name: data.tobytes()})
    finally:
        out.close()


def find_ref_image(ref_dir: Path, stem: str, exts: List[str]) -> Optional[Path]:
    for e in exts:
        p = ref_dir / f"{stem}{e}"
        if p.exists():
            return p
    return None


def crop_center(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = arr.shape
    if th > h or tw > w:
        raise ValueError(f"crop_center 需要 target <= src，但得到 src=({h},{w}) target=({th},{tw})")
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return arr[dh:dh + th, dw:dw + tw]


def crop_topleft(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = arr.shape
    if th > h or tw > w:
        raise ValueError(f"crop_topleft 需要 target <= src，但得到 src=({h},{w}) target=({th},{tw})")
    return arr[:th, :tw]


def resize_depth(arr: np.ndarray, target_hw: Tuple[int, int], mode: str = "nearest") -> np.ndarray:
    """
    用 torch 插值 resize，支持 nearest/bilinear。
    """
    th, tw = target_hw
    t = torch.from_numpy(arr).float()[None, None, ...]  # [1,1,H,W]
    if mode == "nearest":
        out = F.interpolate(t, size=(th, tw), mode="nearest")
    elif mode == "bilinear":
        out = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"未知 resize mode: {mode}")
    return out[0, 0].cpu().numpy().astype(np.float32)


def main(
    gt_depth_dir: str,
    ref_rgb_dir: str,
    out_dir: str,
    gt_channel: str = "R",
    strategy: str = "crop_center",   # crop_center | crop_topleft | resize
    resize_mode: str = "nearest",    # nearest | bilinear
):
    gt_depth_dir = Path(gt_depth_dir)
    ref_rgb_dir = Path(ref_rgb_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    exr_files = sorted([p for p in gt_depth_dir.iterdir() if p.suffix.lower() == ".exr"])
    if not exr_files:
        raise RuntimeError(f"在 {gt_depth_dir} 没找到 .exr 深度文件")

    for exr_path in exr_files:
        stem = exr_path.stem
        ref_path = find_ref_image(ref_rgb_dir, stem, ref_exts)
        if ref_path is None:
            print(f"[Skip] 找不到参考 RGB: {stem}.*")
            continue

        # 目标尺寸来自参考 RGB
        with Image.open(ref_path) as im:
            tw, th = im.size  # PIL 是 (W,H)
        target_hw = (th, tw)

        depth = read_exr_channel(str(exr_path), gt_channel)

        if depth.shape == target_hw:
            out_path = out_dir / exr_path.name
            write_exr_single_channel(str(out_path), depth, channel_name=gt_channel)
            continue

        if strategy == "crop_center":
            if depth.shape[0] < th or depth.shape[1] < tw:
                # 源比目标小就没法裁剪，只能 resize
                aligned = resize_depth(depth, target_hw, mode=resize_mode)
            else:
                aligned = crop_center(depth, target_hw)
        elif strategy == "crop_topleft":
            if depth.shape[0] < th or depth.shape[1] < tw:
                aligned = resize_depth(depth, target_hw, mode=resize_mode)
            else:
                aligned = crop_topleft(depth, target_hw)
        elif strategy == "resize":
            aligned = resize_depth(depth, target_hw, mode=resize_mode)
        else:
            raise ValueError(f"未知 strategy: {strategy}")

        out_path = out_dir / exr_path.name
        write_exr_single_channel(str(out_path), aligned, channel_name=gt_channel)
        print(f"[OK] {exr_path.name}: {depth.shape} -> {aligned.shape}  ({strategy})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Align GT depth EXR to reference RGB size")
    ap.add_argument("--gt_depth_dir", required=True, help="真值深度 EXR 目录，例如 .../images/DepthImage")
    ap.add_argument("--ref_rgb_dir", required=True, help="参考真值 RGB 目录，例如 .../images/ColorImage")
    ap.add_argument("--out_dir", required=True, help="输出目录，例如 .../images/DepthImage_aligned")
    ap.add_argument("--gt_channel", default="R", help="GT 深度通道名，默认 R")
    ap.add_argument("--strategy", default="crop_center", choices=["crop_center", "crop_topleft", "resize"])
    ap.add_argument("--resize_mode", default="nearest", choices=["nearest", "bilinear"])
    args = ap.parse_args()

    main(
        gt_depth_dir=args.gt_depth_dir,
        ref_rgb_dir=args.ref_rgb_dir,
        out_dir=args.out_dir,
        gt_channel=args.gt_channel,
        strategy=args.strategy,
        resize_mode=args.resize_mode,
    )