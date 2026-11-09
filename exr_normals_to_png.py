#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将指定的一张或多张 EXR 法线图转换为 PNG。

支持两种用法：

1. 输出到目录：
python exr_normals_to_png.py input1.exr input2.exr -o output_dir

2. 单张 EXR 输出为指定 PNG 文件：
python exr_normals_to_png.py input.exr -o output.png

示例：
python exr_normals_to_png.py "/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu/V3_Dive3/exp2/V3_Dive3_blue_1/test/ours_30000/renders_normal_exr/0008.exr" -o "/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu/V3_Dive3/exp2/V3_Dive3_blue_1/test/ours_30000/renders_normal_exr/0008.png" --depth-one-color 252,0,33

默认会自动尝试从法线 EXR 路径推导对应深度 EXR：
renders_normal_exr/0008.exr -> renders_depth_exr/0008.exr
并把深度值等于 1 的像素渲染为指定颜色，默认黑色。

指定深度为 1 区域的颜色：
python exr_normals_to_png.py "/path/to/renders_normal_exr/0008.exr" -o "/path/to/0008.png" --depth-one-color 255,0,255

python exr_normals_to_png.py "/path/to/0016.exr" -o "/path/to/0016.png" --input-range -11 --normalize
python exr_normals_to_png.py "/path/to/0016.exr" "/path/to/0017.exr" -o "/path/to/png_dir" --input-range -11 --normalize
"""

import argparse
from pathlib import Path
import numpy as np

# 读取 EXR 依赖
try:
    import OpenEXR
    import Imath
except Exception:
    raise RuntimeError("需要 openexr 与 imath，请先安装：pip install openexr imath")

# 写 PNG 依赖
_USE_IMAGEIO = True
try:
    import imageio.v3 as iio
except Exception:
    _USE_IMAGEIO = False
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("需要 imageio 或 pillow 以写 PNG，请先安装：pip install imageio 或 pip install pillow")


def linear_to_srgb_np(x):
    """
    将线性 RGB 转成 sRGB。
    注意：如果这是给 UE 或其他渲染管线继续作为法线贴图使用，
    通常应加 --keep-linear，避免做 sRGB 伽马。
    """
    a, t = 0.055, 0.0031308
    x = np.clip(x, 0.0, 1.0)
    low = x * 12.92
    high = (1 + a) * np.power(x, 1 / 2.4) - a
    return np.where(x <= t, low, high)


def _pick_rgb_channels(header):
    """
    尽量从 EXR 中选择 R/G/B 或 X/Y/Z 三个通道。
    如果都没有，就退而求其次取前三个通道。
    """
    chans = header["channels"].keys()
    lower = {c.lower(): c for c in chans}

    if all(k in lower for k in ("r", "g", "b")):
        return lower["r"], lower["g"], lower["b"]

    if all(k in lower for k in ("x", "y", "z")):
        return lower["x"], lower["y"], lower["z"]

    ks = list(chans)
    if len(ks) >= 3:
        return ks[0], ks[1], ks[2]

    raise RuntimeError("未找到可用的三个通道构成 RGB")


def read_exr_rgb(path):
    """
    读取 EXR 的三个通道，返回 H x W x 3 的 float32 数组。
    """
    f = OpenEXR.InputFile(str(path))

    try:
        header = f.header()
        dw = header["dataWindow"]

        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        rc, gc, bc = _pick_rgb_channels(header)

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        R = np.frombuffer(f.channel(rc, FLOAT), dtype=np.float32).reshape(H, W)
        G = np.frombuffer(f.channel(gc, FLOAT), dtype=np.float32).reshape(H, W)
        B = np.frombuffer(f.channel(bc, FLOAT), dtype=np.float32).reshape(H, W)

        img = np.stack([R, G, B], axis=-1).astype(np.float32, copy=False)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        return img

    finally:
        f.close()


def _pick_depth_channel(header, preferred="R"):
    """
    尽量从 EXR 中选择一个深度通道。
    """
    chans = list(header.get("channels", {}).keys())
    lower = {c.lower(): c for c in chans}

    for cand in [preferred, "R", "Z", "Depth", "G", "B"]:
        if cand.lower() in lower:
            return lower[cand.lower()]

    if not chans:
        raise RuntimeError("EXR 中不含任何通道")

    return chans[0]


def read_exr_depth(path, channel_hint="R"):
    """
    读取 EXR 的单个深度通道，返回 H x W 的 float32 数组。
    """
    f = OpenEXR.InputFile(str(path))

    try:
        header = f.header()
        dw = header["dataWindow"]

        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        ch = _pick_depth_channel(header, preferred=channel_hint)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        depth = np.frombuffer(f.channel(ch, FLOAT), dtype=np.float32).reshape(H, W)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return depth

    finally:
        f.close()


def _replace_first_case_insensitive(text, old, new):
    """
    只替换第一个大小写不敏感的片段。
    """
    lower = text.lower()
    old_lower = old.lower()
    idx = lower.find(old_lower)

    if idx < 0:
        return None

    return text[:idx] + new + text[idx + len(old):]


def infer_depth_exr_path(normal_exr_path):
    """
    从法线 EXR 路径自动推导对应深度 EXR 路径。

    主要规则：
    renders_normal_exr/0008.exr -> renders_depth_exr/0008.exr
    """
    candidates = []

    parent_name = normal_exr_path.parent.name
    replaced_parent = _replace_first_case_insensitive(parent_name, "normal", "depth")

    if replaced_parent is not None:
        candidates.append(normal_exr_path.parent.with_name(replaced_parent) / normal_exr_path.name)

    path_text = str(normal_exr_path)
    known_replacements = [
        ("renders_normal_exr", "renders_depth_exr"),
        ("normal_exr", "depth_exr"),
        ("normal", "depth"),
    ]

    for old, new in known_replacements:
        replaced = _replace_first_case_insensitive(path_text, old, new)
        if replaced is not None:
            candidates.append(Path(replaced))

    seen = set()

    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def resolve_depth_mask_path(exr_path, depth_exr_arg=None, auto_depth_mask=True):
    """
    解析本次转换要使用的深度 mask EXR。
    """
    if depth_exr_arg:
        p = Path(depth_exr_arg).expanduser().resolve()

        if not p.exists():
            raise FileNotFoundError("指定的深度 EXR 不存在: {}".format(p))

        if not p.is_file():
            raise FileNotFoundError("指定的深度路径不是文件: {}".format(p))

        if p.suffix.lower() != ".exr":
            raise ValueError("指定的深度文件不是 .exr 文件: {}".format(p))

        return p

    if not auto_depth_mask:
        return None

    return infer_depth_exr_path(exr_path)


def parse_rgb_color(value):
    """
    解析 RGB 颜色。支持:
    - 0,0,0 或 255,0,255
    - 0.0,0.0,0.0 到 1.0,1.0,1.0
    - #RRGGBB
    - black/white/red/green/blue/magenta/cyan/yellow
    """
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "yellow": (255, 255, 0),
    }

    text = str(value).strip()
    lower = text.lower()

    if lower in named:
        return named[lower]

    if text.startswith("#") and len(text) == 7:
        try:
            return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            raise argparse.ArgumentTypeError("颜色格式错误: {}".format(value))

    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("颜色需要是 R,G,B，例如 0,0,0 或 255,0,255")

    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("颜色值需要是数字: {}".format(value))

    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [v * 255.0 for v in vals]

    if any(v < 0.0 or v > 255.0 for v in vals):
        raise argparse.ArgumentTypeError("颜色值需要在 0-255 或 0-1 范围内: {}".format(value))

    return tuple(int(round(v)) for v in vals)


def write_png(path, arr_u8):
    """
    写入 RGB PNG。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if _USE_IMAGEIO:
        iio.imwrite(str(path), arr_u8)
    else:
        Image.fromarray(arr_u8, mode="RGB").save(str(path))


def convert_one(
    exr_path,
    png_path,
    input_range="auto",
    normalize=False,
    keep_linear=False,
    depth_mask_path=None,
    depth_channel="R",
    depth_mask_value=1.0,
    depth_mask_tol=1e-6,
    depth_mask_color=(0, 0, 0),
):
    """
    转换单个 EXR 到 PNG。
    """
    img = read_exr_rgb(exr_path)

    if normalize:
        n = np.linalg.norm(img, axis=-1, keepdims=True)
        n = np.clip(n, 1e-8, None)
        img = img / n

    if input_range == "auto":
        if float(np.min(img)) < -1e-6:
            img01 = 0.5 * (img + 1.0)
        else:
            img01 = np.clip(img, 0.0, 1.0)

    elif input_range == "-11":
        img01 = 0.5 * (img + 1.0)

    elif input_range == "01":
        img01 = np.clip(img, 0.0, 1.0)

    else:
        raise ValueError("input_range 仅支持 auto、01、-11")

    img01 = np.clip(img01, 0.0, 1.0)

    if keep_linear:
        vis = img01
    else:
        vis = linear_to_srgb_np(img01)

    u8 = np.clip(vis * 255.0 + 0.5, 0, 255).astype(np.uint8)

    depth_mask_count = 0

    if depth_mask_path is not None:
        depth = read_exr_depth(depth_mask_path, channel_hint=depth_channel)

        if depth.shape != u8.shape[:2]:
            raise ValueError(
                "深度 EXR 尺寸 {} 与法线 EXR 尺寸 {} 不一致: {}".format(
                    depth.shape, u8.shape[:2], depth_mask_path
                )
            )

        mask = np.isfinite(depth) & np.isclose(
            depth,
            float(depth_mask_value),
            atol=float(depth_mask_tol),
            rtol=0.0,
        )

        depth_mask_count = int(np.count_nonzero(mask))

        if depth_mask_count > 0:
            u8[mask] = np.asarray(depth_mask_color, dtype=np.uint8)

    write_png(png_path, u8)
    return depth_mask_count


def collect_exr_files(exr_args):
    """
    收集并检查用户指定的 EXR 文件。
    这里不用 list[Path] 这种 Python 3.9 语法，以兼容旧版本 Python。
    """
    exr_files = []

    for item in exr_args:
        p = Path(item).expanduser().resolve()

        if not p.exists():
            raise FileNotFoundError("EXR 文件不存在: {}".format(p))

        if not p.is_file():
            raise FileNotFoundError("指定路径不是文件: {}".format(p))

        if p.suffix.lower() != ".exr":
            raise ValueError("指定文件不是 .exr 文件: {}".format(p))

        exr_files.append(p)

    # 去重，同时保持输入顺序
    unique = []
    seen = set()

    for p in exr_files:
        key = str(p).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def build_output_paths(exr_list, output_arg):
    """
    根据 -o 参数构造输出路径。

    情况 1：
    输入一张 EXR，-o 是 .png 文件路径，则输出为该 PNG。

    情况 2：
    输入一张或多张 EXR，-o 是目录，则每张 EXR 输出到该目录下，
    文件名保持不变，只把后缀改为 .png。
    """
    out = Path(output_arg).expanduser().resolve()

    # 用户显式指定了一个 PNG 文件路径
    if out.suffix.lower() == ".png":
        if len(exr_list) != 1:
            raise ValueError("当 -o 指定为具体 .png 文件时，只能输入一张 EXR")

        return [(exr_list[0], out)]

    # 否则认为 -o 是输出目录
    output_pairs = []

    for exr_path in exr_list:
        png_path = out / exr_path.with_suffix(".png").name
        output_pairs.append((exr_path, png_path))

    return output_pairs


def main():
    ap = argparse.ArgumentParser(
        description="将指定的一张或多张 EXR 法线图导出为 PNG"
    )

    ap.add_argument(
        "exr_files",
        nargs="+",
        type=str,
        help="一个或多个 EXR 文件路径",
    )

    ap.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="输出目录，或在只输入一张 EXR 时指定具体 PNG 文件路径",
    )

    ap.add_argument(
        "--input-range",
        default="auto",
        choices=["auto", "01", "-11"],
        help="EXR 数值范围判定方式。auto 会根据是否有负值自动判断",
    )

    ap.add_argument(
        "--normalize",
        action="store_true",
        help="先单位化法线向量，再做范围映射",
    )

    ap.add_argument(
        "--keep-linear",
        action="store_true",
        help="保持线性量化，不做 sRGB 伽马转换。用于法线贴图时通常建议开启",
    )

    ap.add_argument(
        "--depth-exr",
        default=None,
        type=str,
        help="手动指定深度 EXR。仅支持单张法线 EXR 输入；不指定时自动推导",
    )

    ap.add_argument(
        "--no-auto-depth-mask",
        action="store_true",
        help="关闭从 renders_normal_exr 自动查找 renders_depth_exr 并覆盖深度值区域的逻辑",
    )

    ap.add_argument(
        "--depth-channel",
        default="R",
        type=str,
        help="读取深度 EXR 时优先使用的通道，默认 R",
    )

    ap.add_argument(
        "--depth-mask-value",
        default=1.0,
        type=float,
        help="需要覆盖颜色的深度值，默认 1.0",
    )

    ap.add_argument(
        "--depth-mask-tol",
        default=1e-6,
        type=float,
        help="判断深度值相等时使用的绝对容差，默认 1e-6",
    )

    ap.add_argument(
        "--depth-one-color",
        "--depth-mask-color",
        dest="depth_mask_color",
        default="0,0,0",
        type=parse_rgb_color,
        help="深度值命中区域输出的 RGB 颜色，默认 0,0,0；支持 255,0,255、#ff00ff、magenta",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划，不实际写文件",
    )

    args = ap.parse_args()

    exr_list = collect_exr_files(args.exr_files)

    if args.depth_exr is not None and len(exr_list) != 1:
        raise ValueError("--depth-exr 只能在输入单张法线 EXR 时使用")

    output_pairs = build_output_paths(exr_list, args.output)

    total = len(output_pairs)
    done = 0

    print("[发现] 指定 EXR 数量 {}".format(total))

    for exr_path, png_path in output_pairs:
        depth_mask_path = resolve_depth_mask_path(
            exr_path=exr_path,
            depth_exr_arg=args.depth_exr,
            auto_depth_mask=not args.no_auto_depth_mask,
        )

        if args.dry_run:
            print("  计划转换: {}  ->  {}".format(exr_path, png_path))
            if depth_mask_path is not None:
                print("    深度mask: {}  value={}  color={}".format(
                    depth_mask_path,
                    args.depth_mask_value,
                    args.depth_mask_color,
                ))
            elif not args.no_auto_depth_mask:
                print("    深度mask: 未找到对应深度 EXR，将按原逻辑转换")
            continue

        try:
            depth_mask_count = convert_one(
                exr_path=exr_path,
                png_path=png_path,
                input_range=args.input_range,
                normalize=args.normalize,
                keep_linear=args.keep_linear,
                depth_mask_path=depth_mask_path,
                depth_channel=args.depth_channel,
                depth_mask_value=args.depth_mask_value,
                depth_mask_tol=args.depth_mask_tol,
                depth_mask_color=args.depth_mask_color,
            )

            done += 1
            if depth_mask_path is not None:
                print("  [完成] {} -> {}  深度mask像素={}".format(
                    exr_path.name, png_path, depth_mask_count
                ))
            else:
                print("  [完成] {} -> {}".format(exr_path.name, png_path))

        except Exception as e:
            print("[失败] {}  原因: {}".format(exr_path, e))

    if args.dry_run:
        print("\n计划处理 {} 个 EXR 文件".format(total))
    else:
        print("\n完成转换。共指定 {} 个 EXR，成功输出 {} 个 PNG".format(total, done))


if __name__ == "__main__":
    main()
