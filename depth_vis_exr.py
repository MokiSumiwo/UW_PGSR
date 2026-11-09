#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将指定的一张或多张 EXR 深度图转换为 PNG，并把输出写入指定目录。

默认参数对齐单文件可视化脚本：
--map equalize  --invert 1  --gamma 0.9  --low_q 1  --high_q 99  --colormap JET

具备断点续转能力：
- 默认跳过已存在且不比源 EXR 旧的 PNG
- 如需重写全部，使用 --overwrite

用法示例：
python depth_vis_exr.py "/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu/oceanEnvironmentPark_2/exp2/oceanEnvironmentPark_2_blue_1/test/ours_30000/renders_depth_exr/0024.exr" -o "/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu/oceanEnvironmentPark_2/exp2/oceanEnvironmentPark_2_blue_1/test/ours_30000/renders_depth_exr/0024.png"
python depth_vis_exr.py "Z:\depth_exr\a.exr" "Z:\depth_exr\b.exr" -o "D:\depth_png"
python depth_vis_exr.py "Z:\depth_exr\a.exr" "Z:\depth_exr\b.exr" -o "D:\depth_png" --dry-run
python depth_vis_exr.py "Z:\depth_exr\a.exr" "Z:\depth_exr\b.exr" -o "D:\depth_png" --overwrite
"""

import argparse
from pathlib import Path
import numpy as np

# 读取 EXR 依赖
try:
    import OpenEXR, Imath
except Exception:
    raise RuntimeError("需要 openexr 与 imath，pip install openexr imath")

# 写 PNG 依赖
_USE_IMAGEIO = True
try:
    import imageio.v3 as iio
except Exception:
    _USE_IMAGEIO = False
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("需要 imageio 或 pillow 以写 PNG")

# OpenCV 用于着色与 CLAHE
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    raise RuntimeError("需要 OpenCV，pip install opencv-python")


# ---------- EXR 读取 ----------
def _pick_depth_channel(header, preferred="R") -> str:
    chans = list(header.get("channels", {}).keys())
    lower = {c.lower(): c for c in chans}

    for cand in [preferred, "Z", "Depth", "G", "B"]:
        if cand.lower() in lower:
            return lower[cand.lower()]

    if not chans:
        raise RuntimeError("EXR 中不含任何通道")

    return chans[0]


def read_exr_depth(path: Path, channel_hint: str = "R") -> np.ndarray:
    f = OpenEXR.InputFile(str(path))
    try:
        header = f.header()
        dw = header["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        ch = _pick_depth_channel(header, preferred=channel_hint)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        arr = np.frombuffer(f.channel(ch, FLOAT), dtype=np.float32).reshape(H, W)
    finally:
        f.close()

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- 工具 ----------
def percentile_clip(x, low_q, high_q):
    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)

    if hi <= lo:
        hi = lo + 1e-6

    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y, lo, hi


def global_equalize_01(x01):
    flat = x01.ravel()
    order = np.argsort(flat)

    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, endpoint=True)

    return ranks.reshape(x01.shape)


# ---------- 映射与着色 ----------
def depth_to_color(
    depth_norm: np.ndarray,
    near_m: float = 0.0,
    far_m: float = 1000.0,
    low_q: float = 1.0,
    high_q: float = 99.0,
    map_mode: str = "equalize",
    gain: float = 10.0,
    gamma: float = 0.9,
    invert_flag: int = 1,
    local_eq: int = 0,
    colormap: str = "JET",
    keep_linear: bool = False,
):
    if not _HAS_CV2:
        raise RuntimeError("需要 opencv-python 以进行着色和可选 CLAHE")

    invert = bool(invert_flag)
    local_eq_bool = bool(local_eq)

    valid = np.isfinite(depth_norm) & (depth_norm > 0)

    depth_m = np.zeros_like(depth_norm, np.float32)
    depth_m[valid] = np.clip(depth_norm[valid] * 1000.0, near_m, far_m)

    depth01 = np.zeros_like(depth_m, np.float32)

    if np.any(valid):
        d = depth_m[valid] / max(far_m, 1e-6)
        d_clip, lo, hi = percentile_clip(d, low_q, high_q)

        if map_mode == "equalize":
            y = global_equalize_01(d_clip)
        elif map_mode == "sigmoid":
            c = np.median(d_clip)
            z = (d_clip - c) / max(hi - lo, 1e-6)
            y = 1.0 / (1.0 + np.exp(-gain * z))
            y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        else:
            y = d_clip

        if local_eq_bool:
            u8 = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            u8 = clahe.apply(u8)
            y = u8.astype(np.float32) / 255.0

        y = np.clip(y, 0.0, 1.0)

        if not keep_linear:
            y = y ** max(gamma, 1e-6)

        if invert:
            y = 1.0 - y

        depth01[valid] = y

    u8 = (depth01 * 255.0 + 0.5).astype(np.uint8)

    cmap = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }.get(colormap.upper(), cv2.COLORMAP_JET)

    color_bgr = cv2.applyColorMap(u8, cmap)

    if np.any(~valid):
        color_bgr[~valid] = (0, 0, 0)

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb


# ---------- 写 PNG ----------
def write_png(path: Path, arr_u8_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if _USE_IMAGEIO:
        iio.imwrite(str(path), arr_u8_rgb)
    else:
        Image.fromarray(arr_u8_rgb, mode="RGB").save(str(path))


# ---------- 断点续转判定 ----------
def png_is_fresh(png_path: Path, exr_path: Path) -> bool:
    """若 PNG 存在、大小大于 0 且 mtime 不早于 EXR，则视为已转好。"""
    try:
        if not png_path.exists():
            return False
        if png_path.stat().st_size <= 0:
            return False
        return png_path.stat().st_mtime >= exr_path.stat().st_mtime
    except Exception:
        return False


# ---------- 单个文件转换 ----------
def convert_one(
    exr_path: Path,
    png_path: Path,
    channel: str = "R",
    near_m: float = 0.0,
    far_m: float = 1000.0,
    low_q: float = 1.0,
    high_q: float = 99.0,
    map_mode: str = "equalize",
    gain: float = 10.0,
    gamma: float = 0.9,
    invert_flag: int = 1,
    local_eq: int = 0,
    colormap: str = "JET",
    keep_linear: bool = False,
):
    depth_norm = read_exr_depth(exr_path, channel_hint=channel)

    color_rgb = depth_to_color(
        depth_norm,
        near_m=near_m,
        far_m=far_m,
        low_q=low_q,
        high_q=high_q,
        map_mode=map_mode,
        gain=gain,
        gamma=gamma,
        invert_flag=invert_flag,
        local_eq=local_eq,
        colormap=colormap,
        keep_linear=keep_linear,
    )

    write_png(png_path, color_rgb)


# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(
        description="将指定的一张或多张 EXR 深度图导出为 PNG"
    )

    ap.add_argument(
        "exr_files",
        nargs="+",
        type=str,
        help="一张或多张 EXR 文件路径，例如 a.exr b.exr",
    )

    ap.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help=(
            "输出路径。单张 EXR 时可以直接指定 PNG 文件路径，例如 0016.png；"
            "多张 EXR 时必须指定输出目录"
        ),
    )

    # 深度映射参数，默认与单文件脚本对齐
    ap.add_argument("--channel", default="R", choices=["R", "G", "B", "Z"])
    ap.add_argument("--near_m", type=float, default=0.0)
    ap.add_argument("--far_m", type=float, default=1000.0)
    ap.add_argument("--low_q", type=float, default=1.0)
    ap.add_argument("--high_q", type=float, default=99.0)
    ap.add_argument("--map", choices=["equalize", "sigmoid", "linear"], default="equalize")
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--invert", type=int, default=1, help="近红远蓝开关，1 开启，0 关闭")
    ap.add_argument("--local_eq", type=int, default=0, help="在深度域做 CLAHE 局部均衡，1 开启")
    ap.add_argument("--colormap", default="JET")
    ap.add_argument("--keep-linear", action="store_true", help="保持线性，不做 γ 校正")

    # 断点续转与覆盖策略
    ap.add_argument("--overwrite", action="store_true", help="强制重写已存在 PNG，不做断点续转")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际写文件")

    args = ap.parse_args()

    # 检查并整理输入 EXR 文件，去除重复路径
    exr_list = []
    seen = set()

    for p in args.exr_files:
        exr_path = Path(p).resolve()

        if exr_path in seen:
            continue

        seen.add(exr_path)

        if not exr_path.exists() or not exr_path.is_file():
            raise FileNotFoundError(f"输入 EXR 不存在或不是文件: {exr_path}")

        if exr_path.suffix.lower() != ".exr":
            raise ValueError(f"输入文件不是 .exr 格式: {exr_path}")

        exr_list.append(exr_path)

    if not exr_list:
        print("[提示] 未提供有效的 EXR 文件")
        return

    output_arg = Path(args.output).resolve()

    # 构造 EXR 到 PNG 的输出映射
    output_pairs = []

    if len(exr_list) == 1:
        exr_path = exr_list[0]

        # 单张 EXR 时：
        # 1. 若 -o 是 .png 文件路径，则直接输出到这个 PNG
        # 2. 若 -o 是目录，则输出到该目录下的同名 PNG
        if output_arg.suffix.lower() == ".png":
            png_path = output_arg
        else:
            png_path = output_arg / exr_path.with_suffix(".png").name

        output_pairs.append((exr_path, png_path))

    else:
        # 多张 EXR 时，-o 必须是目录，避免多个输入争抢同一个 PNG 文件
        if output_arg.suffix.lower() == ".png":
            raise ValueError(
                "指定多张 EXR 时，-o/--output 必须是输出目录，不能是单个 PNG 文件路径"
            )

        out_dir = output_arg

        # 检查不同目录下同名 EXR 导致输出 PNG 重名的问题
        output_map = {}

        for exr_path in exr_list:
            png_path = out_dir / exr_path.with_suffix(".png").name

            if png_path in output_map and output_map[png_path] != exr_path:
                raise RuntimeError(
                    "存在同名 EXR 会输出到同一个 PNG，请重命名输入文件或分开输出:\n"
                    f"  {output_map[png_path]}\n"
                    f"  {exr_path}\n"
                    f"  输出目标: {png_path}"
                )

            output_map[png_path] = exr_path
            output_pairs.append((exr_path, png_path))

    total = len(output_pairs)
    done = 0
    skipped = 0

    print(f"[发现] 指定 EXR 数量 {total}")

    for exr_path, png_path in output_pairs:
        # 注意：这里创建的是 PNG 所在的父目录，而不是 PNG 自身
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # 断点续转判定
        if not args.overwrite and png_is_fresh(png_path, exr_path):
            if args.dry_run:
                print(f"  [已存在] 跳过  {exr_path}  ->  {png_path}")
            else:
                print(f"  [已存在] 跳过  {png_path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  计划转换: {exr_path}  ->  {png_path}")
            continue

        try:
            convert_one(
                exr_path,
                png_path,
                channel=args.channel,
                near_m=args.near_m,
                far_m=args.far_m,
                low_q=args.low_q,
                high_q=args.high_q,
                map_mode=args.map,
                gain=args.gain,
                gamma=args.gamma,
                invert_flag=args.invert,
                local_eq=args.local_eq,
                colormap=args.colormap,
                keep_linear=args.keep_linear,
            )

            done += 1
            print(f"  [完成] {exr_path} -> {png_path}")

        except Exception as e:
            print(f"[失败] {exr_path}  原因: {e}")

    if args.dry_run:
        print(
            f"\n计划处理 {total} 个 EXR 文件，其中将跳过 {skipped} 个已完成，"
            f"预计转换 {total - skipped} 个"
        )
    else:
        print(
            f"\n完成转换。共指定 {total} 个 EXR，跳过 {skipped} 个已完成，"
            f"成功输出 {done} 个 PNG"
        )


if __name__ == "__main__":
    main()