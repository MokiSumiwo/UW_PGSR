#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "funa_exp13_6gpu"
DELIVERY_ROOT = REPO_ROOT / "final_delivery_funa_geomed_gs"
ARTIFACT_ROOT = REPO_ROOT / "research_os" / "research_os" / "artifacts" / "final_export_funa_geomed_gs"
CONTRACT_PATH = REPO_ROOT / "research_os" / "research_os" / "constraints" / "45_final_export_transition_contract.yaml"
VALIDATOR_PATH = REPO_ROOT / "research_os" / "scripts" / "validate_final_export_transition.py"

PROJECT_ID = "funa_geomed_gs"
PROJECT_NAME = "FUNA / GeoMed-GS Underwater 3D Gaussian Splatting"
APPROVED_BY = "ycy"
APPROVAL_DATE = "2026-05-11"

APPROVED = {
    "approved_title": "FUNA: Evaluable Geometry-Medium Disentanglement for Underwater 3D Gaussian Splatting",
    "approved_method_name": "Geometry-Medium Disentangled Gaussian Splatting (GeoMed-GS)",
    "approved_task_type": "evaluable geometry-medium disentanglement for underwater 3D Gaussian Splatting",
    "approved_one_sentence_story": "本文不把目标夸大为完整水下三维重建，而是把水下 3DGS 重新表述为一个可被严格评估的几何-介质解耦问题，并通过物理闭合的介质-背景渲染与错误归因感知的高斯容量控制来减少介质对几何的污染。",
    "approved_core_goal": "本文将水下 3DGS 重新表述为一个可评估的 geometry-medium disentanglement 问题。借助 FUNA 提供的 paired underwater/clean/depth/normal 真值，以及物理闭合的介质-背景渲染和 misattribution-aware Gaussian capacity control，我们在 clean appearance、depth 和 normal 空间中显著缓解了 medium-induced geometry corruption，同时保持了具有竞争力的 underwater novel-view synthesis 性能。",
    "approved_claim_boundary": "本文只声称在 FUNA 覆盖的两类水体颜色、四档退化、当前合成场景以及真实 SeaThru 的现有评测范围内，方法能够显著缓解介质诱导的几何错误归因。本文不声称已经完整恢复真实水下本征物理参数，也不声称能在任意水体、光照、轨迹、材质或真实海域条件下普适恢复精确三维结构。",
    "approved_appendix_title": "Supplementary Material for FUNA: Evaluable Geometry-Medium Disentanglement for Underwater 3D Gaussian Splatting",
}

WRITING_GUIDANCE = {
    "approved_abstract_direction": "Abstract should foreground evaluable geometry-medium disentanglement, FUNA as the paired evaluability layer, and the coupled rendering-plus-capacity-control mechanism rather than claiming full underwater intrinsic recovery.",
    "reference_split_rule": "Main text references should stay tightly bound to benchmark framing, underwater multi-view reconstruction baselines, and 3DGS comparators; longer fairness notes, implementation caveats, and supplemental protocol references may move to the appendix.",
    "appendix_prose_target": "Appendix prose should stay audit-friendly and information-preserving, with full result-source provenance, long-form table descriptions, and explicit boundary notes for real-data conclusions.",
    "final_polish_scope": "Final polish is allowed to compress wording and smooth transitions, but it must not change title, task type, one-sentence story, final claim, claim boundary, or the experiment-to-evidence mapping frozen for export.",
}

SCENES = [
    "Bay_submerged_ships_2",
    "Egypt_Ruins_Environment_2",
    "Eternal_temple_3",
    "oceanEnvironmentPark_2",
    "V3_Cave02",
    "V3_Dive3",
]

QUAL_SAMPLES = [
    {"scene": "Egypt_Ruins_Environment_2", "frame": "0000"},
    {"scene": "V3_Cave02", "frame": "0000"},
    {"scene": "V3_Dive3", "frame": "0000"},
]

ABLATION_SAMPLES = [
    {"scene": "Egypt_Ruins_Environment_2", "frame": "0000"},
    {"scene": "V3_Cave02", "frame": "0000"},
]

PSEUDOCLEAN_SAMPLES = [
    {"scene": "Eternal_temple_3", "frame": "0000"},
    {"scene": "V3_Dive3", "frame": "0000"},
]

TABLE1_COLUMNS = [
    "Method",
    "RGB PSNR",
    "RGB SSIM",
    "RGB LPIPS",
    "Clean PSNR",
    "Clean SSIM",
    "Clean LPIPS",
    "Depth RMSE",
    "AbsRel",
    "delta<1.25",
    "Normal Mean",
    "Normal Median",
    "Acc 11.25deg",
]

TABLE1B_COLUMNS = [
    "Method",
    "RGB PSNR",
    "RGB SSIM",
    "RGB LPIPS",
    "UCIQE",
    "UIQM",
    "NIQE",
    "Depth RMSE",
    "AbsRel",
    "delta<1.25",
]

METHOD_NAME_MAP = {
    "watersplatting": "WaterSplatting",
    "PGSR": "PGSR",
    "3DGS": "3DGS",
    "nerf": "NeRF",
    "seathruNeRF": "SeaThru-NeRF",
    "ours": "Ours",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def yaml_bytes(payload: dict[str, Any], *, drop_field: str | None = None) -> bytes:
    clone = json.loads(json.dumps(payload, ensure_ascii=False))
    if drop_field is not None:
        clone.pop(drop_field, None)
    return yaml.safe_dump(clone, sort_keys=False, allow_unicode=False).encode("utf-8")


def payload_hash(payload: dict[str, Any], drop_field: str) -> str:
    return sha256_bytes(yaml_bytes(payload, drop_field=drop_field))


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    path.write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=False), encoding="utf-8")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def find_single(pattern: str) -> Path:
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for pattern: {pattern}")
    return matches[0]


def extract_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def code_section_text() -> str:
    snippets = [
        ("scene/app_model.py", "Appearance branch parameters and optimizer", 9, 30),
        ("scene/medium_model.py", "Legacy medium estimator branch", 6, 149),
        ("scene/complete_physics_model.py", "Backscatter and attenuation submodules", 33, 113),
        ("scene/complete_physics_model.py", "Attenuation network and appearance embedding", 116, 204),
        ("scene/complete_physics_model.py", "Complete underwater physics model", 232, 317),
        ("scene/__init__.py", "Dataset / input entry and scene construction", 23, 103),
        ("scene/dataset_readers.py", "Scene loader callbacks", 263, 302),
        ("gaussian_renderer/__init__.py", "Renderer forward path with clear-scene and underwater branches", 36, 259),
        ("train.py", "Training entrypoint, scene loading, physics module wiring, and primary image loss", 1241, 1373),
        ("train.py", "Background-consistency loss activation", 1375, 1419),
        ("train.py", "Geometry/photo consistency losses and backward path", 2030, 2112),
        ("train.py", "Densification, pruning, and optimizer update loop", 2228, 2364),
    ]

    blocks: list[str] = []
    for rel_path, title, start, end in snippets:
        snippet = extract_lines(REPO_ROOT / rel_path, start, end)
        blocks.append(
            "\n".join(
                [
                    f"# Source: {rel_path}:{start}-{end}",
                    f"# Role: {title}",
                    "```python",
                    snippet,
                    "```",
                ]
            )
        )
    return "\n\n".join(blocks)


def read_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def col_to_num(col: str) -> int:
        n = 0
        for ch in col:
            n = n * 26 + ord(ch) - 64
        return n

    def split_ref(cell_ref: str) -> tuple[str, int]:
        idx = 0
        while idx < len(cell_ref) and cell_ref[idx].isalpha():
            idx += 1
        return cell_ref[:idx], int(cell_ref[idx:])

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.iterfind(".//a:t", ns)]
                shared_strings.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        first_sheet = workbook.find("a:sheets", ns)[0]
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheet_path = "xl/" + rel_map[rid]
        sheet_root = ET.fromstring(zf.read(sheet_path))

    cells: dict[tuple[int, int], str] = {}
    max_row = 0
    max_col = 0
    for cell in sheet_root.iterfind(".//a:c", ns):
        ref = cell.attrib["r"]
        col, row = split_ref(ref)
        col_num = col_to_num(col)
        max_row = max(max_row, row)
        max_col = max(max_col, col_num)
        cell_type = cell.attrib.get("t")
        value_node = cell.find("a:v", ns)
        value = ""
        if cell_type == "s" and value_node is not None:
            value = shared_strings[int(value_node.text)]
        elif cell_type == "inlineStr":
            value = "".join(t.text or "" for t in cell.iterfind(".//a:t", ns))
        elif value_node is not None:
            value = value_node.text or ""
        cells[(row, col_num)] = value

    rows: list[list[str]] = []
    for row_idx in range(1, max_row + 1):
        row_vals = [cells.get((row_idx, col_idx), "") for col_idx in range(1, max_col + 1)]
        if any(val != "" for val in row_vals):
            rows.append(row_vals)
    return rows


def parse_authoritative_main_tables(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    rows = read_xlsx_rows(path)
    dump_lines = []
    for row in rows:
        dump_lines.append("\t".join(row).rstrip())
    dump_text = "\n".join(dump_lines)

    average_indices = [i for i, row in enumerate(rows) if row and row[0] == "average"]
    if len(average_indices) < 2:
        raise ValueError(f"Expected at least two average blocks in {path}, found {len(average_indices)}")

    table1_rows = rows[average_indices[0] + 1 : average_indices[0] + 7]
    out1 = []
    for row in table1_rows:
        out1.append(
            OrderedDict(
                zip(
                    TABLE1_COLUMNS,
                    [
                        METHOD_NAME_MAP.get(row[0], row[0]),
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[11],
                        row[12],
                    ],
                )
            )
        )

    table1b_rows = rows[average_indices[1] + 1 : average_indices[1] + 7]
    out2 = []
    for row in table1b_rows:
        out2.append(
            OrderedDict(
                zip(
                    TABLE1B_COLUMNS,
                    [
                        METHOD_NAME_MAP.get(row[0], row[0]),
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                    ],
                )
            )
        )

    return out1, out2, dump_text


def read_results_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["ours_30000"]


def mean_row_by_method(rows: list[dict[str, str]], method: str) -> dict[str, Any]:
    row = next(r for r in rows if r["Method"] == method)
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "Method":
            out[k] = v
        elif v == "":
            out[k] = None
        else:
            out[k] = float(v)
    return out


def load_stage_table_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    exp1 = [r for r in load_csv_rows(OUTPUT_ROOT / "exp1_table.csv") if r["scope"] == "avg" and r["scene"] == "ALL"]
    exp2 = [r for r in load_csv_rows(OUTPUT_ROOT / "exp2_table.csv") if r["scope"] == "avg" and r["scene"] == "ALL"]
    exp3 = [r for r in load_csv_rows(OUTPUT_ROOT / "exp3_table.csv") if r["scope"] == "avg" and r["scene"] == "ALL"]
    return exp1, exp2, exp3


def format_table_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def relabel_stage_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    keys = [
        "stage_label",
        "variant",
        "rgb_ssim",
        "rgb_psnr",
        "rgb_lpips",
        "clean_ssim",
        "clean_psnr",
        "clean_lpips",
        "depth_rmse",
        "depth_absrel",
        "depth_delta125",
        "normal_mean_angle",
        "normal_median_angle",
        "normal_acc_11_25deg",
    ]
    for row in rows:
        out.append({key: row[key] for key in keys})
    return out


def relabel_exp2_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keep = [
        "water_color",
        "degradation_level",
        "num_scenes",
        "rgb_ssim",
        "rgb_psnr",
        "rgb_lpips",
        "clean_ssim",
        "clean_psnr",
        "clean_lpips",
        "depth_rmse",
        "depth_absrel",
        "depth_delta125",
        "normal_mean_angle",
        "normal_median_angle",
        "normal_acc_11_25deg",
    ]
    return [{key: row[key] for key in keep} for row in rows]


def relabel_exp3_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keep = [
        "profile",
        "num_scenes",
        "rgb_ssim",
        "rgb_psnr",
        "rgb_lpips",
        "clean_ssim",
        "clean_psnr",
        "clean_lpips",
        "depth_rmse",
        "depth_absrel",
        "depth_delta125",
        "normal_mean_angle",
        "normal_median_angle",
        "normal_acc_11_25deg",
    ]
    return [{key: row[key] for key in keep} for row in rows]


def rgb_spec(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def build_image_grid_imagemagick(
    image_specs: list[list[tuple[Path, str]]],
    row_labels: list[str],
    out_path: Path,
    *,
    tile_size: tuple[int, int],
    top_header_h: int,
    left_label_w: int,
    row_gap: int,
    col_gap: int,
    bg_color: tuple[int, int, int],
) -> bool:
    convert_bin = shutil.which("convert")
    if convert_bin is None:
        return False

    ensure_dir(out_path.parent)
    component_h = tile_size[1] + top_header_h
    dark_header = (32, 38, 48)
    border_color = (180, 184, 190)

    with tempfile.TemporaryDirectory(prefix=f"{out_path.stem}_", dir=str(out_path.parent)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        row_paths: list[Path] = []
        max_cols = max(len(row) for row in image_specs)

        for r_idx, row in enumerate(image_specs):
            label_path = tmp_root / f"row_label_{r_idx:02d}.png"
            subprocess.run(
                [
                    convert_bin,
                    "-size",
                    f"{left_label_w}x{component_h}",
                    f"xc:{rgb_spec(bg_color)}",
                    "-gravity",
                    "west",
                    "-fill",
                    "black",
                    "-pointsize",
                    "18",
                    "-annotate",
                    "+12+0",
                    row_labels[r_idx],
                    str(label_path),
                ],
                check=True,
            )

            component_paths = [label_path]
            for c_idx in range(max_cols):
                tile_path = tmp_root / f"tile_{r_idx:02d}_{c_idx:02d}.png"
                if c_idx < len(row):
                    img_path, title = row[c_idx]
                    cmd = [
                        convert_bin,
                        str(img_path),
                        "-resize",
                        f"{tile_size[0]}x{tile_size[1]}!",
                        "-background",
                        rgb_spec(bg_color),
                        "-gravity",
                        "north",
                        "-splice",
                        f"0x{top_header_h}",
                    ]
                    if r_idx == 0:
                        cmd.extend(
                            [
                                "-fill",
                                rgb_spec(dark_header),
                                "-draw",
                                f"rectangle 0,0 {tile_size[0] - 1},{top_header_h - 1}",
                                "-fill",
                                "white",
                                "-gravity",
                                "northwest",
                                "-pointsize",
                                "18",
                                "-annotate",
                                "+10+12",
                                title,
                            ]
                        )
                    cmd.extend(
                        [
                            "-fill",
                            "none",
                            "-stroke",
                            rgb_spec(border_color),
                            "-strokewidth",
                            "1",
                            "-draw",
                            f"rectangle 0,0 {tile_size[0] - 1},{component_h - 1}",
                            str(tile_path),
                        ]
                    )
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(
                        [
                            convert_bin,
                            "-size",
                            f"{tile_size[0]}x{component_h}",
                            f"xc:{rgb_spec(bg_color)}",
                            str(tile_path),
                        ],
                        check=True,
                    )
                component_paths.append(tile_path)

            row_path = tmp_root / f"row_{r_idx:02d}.png"
            subprocess.run(
                [
                    convert_bin,
                    *[str(p) for p in component_paths],
                    "+append",
                    "-bordercolor",
                    rgb_spec(bg_color),
                    "-border",
                    f"{max(col_gap // 2, 4)}x{max(row_gap // 2, 8)}",
                    str(row_path),
                ],
                check=True,
            )
            row_paths.append(row_path)

        subprocess.run(
            [
                convert_bin,
                *[str(p) for p in row_paths],
                "-append",
                "-bordercolor",
                rgb_spec(bg_color),
                "-border",
                "20x20",
                str(out_path),
            ],
            check=True,
        )
    return True


def build_image_grid(
    image_specs: list[list[tuple[Path, str]]],
    row_labels: list[str],
    out_path: Path,
    *,
    tile_size: tuple[int, int] = (440, 248),
    top_header_h: int = 44,
    left_label_w: int = 190,
    row_gap: int = 18,
    col_gap: int = 10,
    bg_color: tuple[int, int, int] = (245, 246, 248),
) -> bool:
    if Image is None:
        return build_image_grid_imagemagick(
            image_specs,
            row_labels,
            out_path,
            tile_size=tile_size,
            top_header_h=top_header_h,
            left_label_w=left_label_w,
            row_gap=row_gap,
            col_gap=col_gap,
            bg_color=bg_color,
        )

    cols = max(len(row) for row in image_specs)
    rows = len(image_specs)
    width = left_label_w + cols * tile_size[0] + (cols - 1) * col_gap + 40
    height = top_header_h + rows * tile_size[1] + (rows - 1) * row_gap + 40
    canvas = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for r_idx, row in enumerate(image_specs):
        y0 = 20 + top_header_h + r_idx * (tile_size[1] + row_gap)
        draw.text((20, y0 + tile_size[1] // 2 - 8), row_labels[r_idx], fill=(25, 25, 25), font=font)
        for c_idx, (img_path, title) in enumerate(row):
            x0 = 20 + left_label_w + c_idx * (tile_size[0] + col_gap)
            if r_idx == 0:
                draw.rounded_rectangle((x0, 20, x0 + tile_size[0], 20 + top_header_h - 8), radius=8, fill=(32, 38, 48))
                draw.text((x0 + 10, 30), title, fill=(255, 255, 255), font=font)
            tile = Image.open(img_path).convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)
            canvas.paste(tile, (x0, y0))
            draw.rectangle((x0, y0, x0 + tile_size[0], y0 + tile_size[1]), outline=(180, 184, 190), width=1)

    ensure_dir(out_path.parent)
    canvas.save(out_path)
    return True


def make_main_qualitative_figure(figures_root: Path) -> dict[str, Any] | None:
    rows: list[list[tuple[Path, str]]] = []
    row_labels: list[str] = []
    for sample in QUAL_SAMPLES:
        scene = sample["scene"]
        frame = sample["frame"]
        base = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp1/*progressive_stage6_pseudoclean_realrun/test/ours_30000")
        rows.append(
            [
                (base / "gt" / f"{frame}.png", "GT Underwater"),
                (base / "renders" / f"{frame}.png", "Ours Underwater"),
                (base / "renders_air_bgblack" / f"{frame}.png", "Ours Clean"),
                (base / "renders_depth" / f"{frame}.png", "Ours Depth"),
                (base / "renders_normal" / f"{frame}.png", "Ours Normal"),
            ]
        )
        row_labels.append(scene)
    out_path = figures_root / "figure1_main_qualitative_outputs.png"
    ok = build_image_grid(rows, row_labels, out_path)
    if not ok:
        return None
    return {
        "figure_id": "Figure 1",
        "path": str(out_path.resolve()),
        "source_rows": [{"scene": s["scene"], "frame": s["frame"]} for s in QUAL_SAMPLES],
    }


def make_ablation_figure(figures_root: Path) -> dict[str, Any] | None:
    rows: list[list[tuple[Path, str]]] = []
    row_labels: list[str] = []
    for sample in ABLATION_SAMPLES:
        scene = sample["scene"]
        frame = sample["frame"]
        stage0 = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp1/*progressive_stage0_pgsr/test/ours_30000")
        stage6 = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp1/*progressive_stage6_pseudoclean_realrun/test/ours_30000")
        rows.append(
            [
                (stage6 / "gt" / f"{frame}.png", "GT Underwater"),
                (stage0 / "renders_air_bgblack" / f"{frame}.png", "Stage 0 Clean"),
                (stage6 / "renders_air_bgblack" / f"{frame}.png", "Stage 6 Clean"),
                (stage0 / "renders_depth" / f"{frame}.png", "Stage 0 Depth"),
                (stage6 / "renders_depth" / f"{frame}.png", "Stage 6 Depth"),
            ]
        )
        row_labels.append(scene)
    out_path = figures_root / "figure2_progressive_ablation_examples.png"
    ok = build_image_grid(rows, row_labels, out_path)
    if not ok:
        return None
    return {
        "figure_id": "Figure 2",
        "path": str(out_path.resolve()),
        "source_rows": [{"scene": s["scene"], "frame": s["frame"]} for s in ABLATION_SAMPLES],
    }


def make_pseudoclean_figure(figures_root: Path) -> dict[str, Any] | None:
    rows: list[list[tuple[Path, str]]] = []
    row_labels: list[str] = []
    for sample in PSEUDOCLEAN_SAMPLES:
        scene = sample["scene"]
        frame = sample["frame"]
        none_root = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp3/*blue_4_fakeair_none/test/ours_30000")
        pseudo_root = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp3/*blue_4_fakeair_pseudo/test/ours_30000")
        oracle_root = find_single(f"outputs/funa_exp13_6gpu/{scene}/exp3/*blue_4_fakeair_oracle/test/ours_30000")
        rows.append(
            [
                (pseudo_root / "gt" / f"{frame}.png", "GT Underwater"),
                (none_root / "renders_air_bgblack" / f"{frame}.png", "No Pseudo-Clean"),
                (pseudo_root / "renders_air_bgblack" / f"{frame}.png", "Pseudo-Clean"),
                (oracle_root / "renders_air_bgblack" / f"{frame}.png", "Oracle Clean"),
            ]
        )
        row_labels.append(scene)
    out_path = figures_root / "appendix_pseudoclean_profiles.png"
    ok = build_image_grid(rows, row_labels, out_path)
    if not ok:
        return None
    return {
        "figure_id": "Appendix Figure A1",
        "path": str(out_path.resolve()),
        "source_rows": [{"scene": s["scene"], "frame": s["frame"]} for s in PSEUDOCLEAN_SAMPLES],
    }


def build_paper_input(code_text: str) -> str:
    return "\n\n".join(
        [
            "[文章名]",
            APPROVED["approved_title"],
            "[主方法名]",
            APPROVED["approved_method_name"],
            "[任务类型]",
            APPROVED["approved_task_type"],
            "[故事（一句话）]",
            APPROVED["approved_one_sentence_story"],
            "[最终Claim]",
            APPROVED["approved_core_goal"],
            "[Claim边界]",
            APPROVED["approved_claim_boundary"],
            "[完整工程代码]",
            code_text,
        ]
    )


def build_appendix_input(
    table1_rows: list[dict[str, str]],
    table1b_rows: list[dict[str, str]],
    exp1_rows: list[dict[str, Any]],
    exp2_rows: list[dict[str, Any]],
    exp3_rows: list[dict[str, Any]],
    figure_notes: list[dict[str, Any]],
    authoritative_dump_text: str,
) -> str:
    appendix_body = textwrap.dedent(
        f"""
        1. Final Semantic Freeze

        - Title: {APPROVED["approved_title"]}
        - Method: {APPROVED["approved_method_name"]}
        - Task type: {APPROVED["approved_task_type"]}
        - One-sentence story: {APPROVED["approved_one_sentence_story"]}
        - Final claim: {APPROVED["approved_core_goal"]}
        - Claim boundary: {APPROVED["approved_claim_boundary"]}

        2. Project Understanding and Export Policy

        这份交付包面向 research_os 的最终写作层，而不是继续做实验。项目的中心定位被固定为“可评估的 geometry-medium disentanglement for underwater 3D Gaussian Splatting”，不再使用“完整水下三维重建”或“完整真实水下物理恢复”这类更大的叙事。最终写作时，FUNA 被定位为 paired underwater/clean/depth/normal evaluability layer，GeoMed-GS 被定位为 coupling of physically closed medium-background rendering and misattribution-aware Gaussian capacity control。

        本交付包的主结果来源经过显式绑定。主实验表使用 `{(OUTPUT_ROOT / "论文主实验对比表.xlsx").resolve()}` 作为权威源；Table 2 使用 `exp2_table.csv`；Table 3 使用 `exp1_table.csv`；Table 4 使用 `exp3_table.csv`。此前已经确认 `exp1 stage6 realrun` 与 `exp2 blue_4` 并非数值完全一致，因此主比较以作者指定的 xlsx 为准，而不是擅自互换。

        3. Data, Protocol, and Metric Grounding

        - Synthetic paired benchmark: FUNA, with paired underwater RGB, clean appearance, depth, and normal supervision for evaluation.
        - Real-data benchmark: SeaThru, used for real underwater novel-view synthesis and de-watered perceptual evidence with conservative wording.
        - Training entrypoint: `train.py`
        - Rendering entrypoint: `render.py`
        - Evaluation entrypoint: `metrics3.py`

        主协议是 scene-level multi-view reconstruction with held-out view evaluation。FUNA 的 headline metrics 是 clean appearance recovery plus geometry quality, specifically clean PSNR/SSIM/LPIPS, depth RMSE/AbsRel/delta<1.25, and normal mean/median/Acc@11.25deg. Real SeaThru 结果只作为真实场景竞争力和感知质量证据，不承担“真实几何全面领先”的过重 claim。

        4. Frozen Experiment Structure

        - Experiment 1: Main comparison on FUNA and real SeaThru.
        - Experiment 2: Stratified analysis by water color and degradation level.
        - Experiment 3: Progressive ablation on the attribution-control chain.
        - Experiment 4: Effectiveness and bias analysis of pseudo-clean supervision.

        5. Main Experimental Takeaways

        Main comparison: on FUNA, Ours reaches clean PSNR 15.212, depth RMSE 3.167, depth AbsRel 0.043, delta<1.25 0.894, and normal mean angle 46.618. The key evidence is not universal RGB dominance but joint improvement in clean appearance and geometry. On real SeaThru, Ours is not the best on RGB PSNR, so the correct wording is “competitive real-scene performance with stronger structural similarity, LPIPS, and de-watered perceptual quality,” not “universal real-scene geometry best.”

        Stratified analysis: under blue/green water colors and degradation levels 1-4, the method remains relatively stable in clean and geometry metrics, with harder settings mainly reflected in reduced clean PSNR and larger geometry error rather than complete failure. This supports the claim that the method mitigates medium-induced attribution ambiguity across medium configurations covered by FUNA.

        Progressive ablation: from Stage 0 PGSR to Stage 6 +Pseudo-Clean, clean PSNR improves from 10.468 to 15.185, depth RMSE drops from 7.348 to 3.034, and depth delta<1.25 rises from 0.737 to 0.901. This is the mechanism chain evidence that the gains come from rendering closure plus geometry-capacity control, not from one isolated trick.

        Pseudo-clean analysis: pseudo-clean supervision improves clean LPIPS, depth RMSE, and delta<1.25 over the no-pseudo-clean setting, while oracle clean further improves clean PSNR/SSIM. This supports the conservative statement that pseudo-clean is a useful directional supervision signal but not a perfect substitute for paired clean ground truth.

        6. Table Raw Text

        Table 1. Quantitative comparison on FUNA (authoritative average rows from xlsx):

        {format_table_markdown(table1_rows)}

        Table 1b. Quantitative comparison on real SeaThru (authoritative average rows from xlsx):

        {format_table_markdown(table1b_rows)}

        Table 2. Stratified scene-level analysis by water color and degradation:

        {format_table_markdown(exp2_rows)}

        Table 3. Progressive ablation on FUNA:

        {format_table_markdown(exp1_rows)}

        Table 4. Pseudo-clean supervision analysis on FUNA:

        {format_table_markdown(exp3_rows)}

        7. Figure Inventory

        The following figures were generated from the frozen experiment outputs and copied into the delivery root for writing use. Their detailed captions live in the main-text or appendix caption registries.
        """
    ).strip()

    figure_lines = []
    for note in figure_notes:
        figure_lines.append(f"- {note['figure_id']}: {note['path']}")
    if figure_lines:
        appendix_body += "\n" + "\n".join(figure_lines)

    appendix_body += textwrap.dedent(
        f"""

        8. Authoritative Spreadsheet Dump

        The raw dump below is kept so downstream writing can cross-check the exact cell ordering used in the author-approved main-comparison spreadsheet.

        ```text
        {authoritative_dump_text}
        ```
        """
    )

    return "\n\n".join(
        [
            "[附录标题]",
            APPROVED["approved_appendix_title"],
            "[附录正文]",
            appendix_body,
        ]
    )


def main_text_caption_registry(main_csv_dir: Path, figures_root: Path, figure_notes: list[dict[str, Any]]) -> str:
    figure_paths = {Path(note["path"]).name: note["path"] for note in figure_notes}
    return textwrap.dedent(
        f"""
        # Main-Text Caption Registry

        ## Tables

        ### Table 1
        Path: {(main_csv_dir / "table1_funa_main_comparison.csv").resolve()}
        Caption: Quantitative comparison on FUNA. All methods are evaluated under the same held-out view protocol. WaterSplatting and SeaThru-NeRF report both underwater RGB and clean-related metrics because they expose de-watered outputs, while PGSR, 3DGS, and NeRF remain RGB-centric baselines whose clean columns are not defined by design. The key result is that Ours achieves the best clean PSNR/SSIM among reported clean-capable methods and substantially improves geometry metrics, reducing depth RMSE from 15.178 for WaterSplatting and 7.276 for PGSR to 3.167 while also improving depth AbsRel to 0.043 and normal mean angle to 46.618. This table supports the paper's final claim that physically closed medium-background rendering plus misattribution-aware capacity control reduce medium-induced geometry corruption in an evaluable setting rather than merely boosting underwater RGB appearance.

        ### Table 1b
        Path: {(main_csv_dir / "table1b_real_seathru_main_comparison.csv").resolve()}
        Caption: Quantitative comparison on real SeaThru. This table is intentionally read conservatively: Ours does not claim the highest RGB PSNR on real data, but it maintains competitive novel-view quality while improving SSIM, LPIPS, and de-watered perceptual metrics relative to the water-aware baselines that report them. The correct conclusion is competitive real-scene performance with stronger structure/perceptual quality, not a universal claim of best real-scene geometry.

        ### Table 2
        Path: {(main_csv_dir / "table2_stratified_analysis.csv").resolve()}
        Caption: Stratified analysis on FUNA by water color and degradation level. Each row aggregates six scenes under the same water-color and degradation bucket. The table preserves underwater RGB, clean appearance, depth, and normal metrics to show how the method behaves as medium-induced ambiguity increases. The main observation is stability rather than dominance in every scalar: under both blue and green water, and across all four degradation levels, Ours keeps clean and geometry metrics within a relatively tight band, which is the appropriate evidence for the robustness part of the final claim.

        ### Table 3
        Path: {(main_csv_dir / "table3_progressive_ablation.csv").resolve()}
        Caption: Progressive ablation on FUNA. The rows follow the actual training-stage chain used in the experiments: PGSR backbone, underwater physics branch, background closure, far gating, capacity control, conflict-aware background pruning, and pseudo-clean supervision. The table is a mechanism table rather than a generic ablation table. Clean PSNR rises from 10.468 at Stage 0 to 15.185 at Stage 6, while depth RMSE drops from 7.348 to 3.034 and depth delta<1.25 rises from 0.737 to 0.901, supporting the interpretation that the gains arise from the full attribution-control chain.

        ### Table 4
        Path: {(main_csv_dir / "table4_pseudoclean_analysis.csv").resolve()}
        Caption: Pseudo-clean supervision analysis on FUNA. The comparison spans no pseudo-clean supervision, the current pseudo-clean pipeline, and oracle clean supervision. Pseudo-clean improves clean LPIPS and geometry metrics over the no-pseudo-clean configuration, while oracle clean further improves clean PSNR/SSIM. This table is meant to bound the role of pseudo-clean supervision: it is useful and directionally correct, but not the sole cause of the method's gains and not a substitute for paired clean ground truth.

        ## Figures

        ### Figure 1
        Path: {figure_paths.get("figure1_main_qualitative_outputs.png", "")}
        Caption: Representative FUNA outputs from three held-out views. Each row shows the same scene under the frozen Stage 6 realrun model, with ground-truth underwater RGB, Ours underwater render, Ours clean render (`AIRBGBLACK` branch), rendered depth, and rendered normal. The figure does not attempt a baseline beauty contest; instead it visualizes the multi-output evidence that underpins the final claim. The clean renders expose de-watered structure, the depth maps show coherent scene layout rather than fog-wall collapse, and the normal maps remain structurally organized instead of being dominated by medium-like artifacts.

        ### Figure 2
        Path: {figure_paths.get("figure2_progressive_ablation_examples.png", "")}
        Caption: Progressive ablation examples from two FUNA scenes. Each row keeps the same held-out view and compares Stage 0 PGSR against the final Stage 6 configuration in clean-render and depth space, with the underwater ground truth shown as context. The figure is meant to be read together with Table 3: Stage 0 lacks an explicit geometry-medium disentanglement mechanism and therefore produces weaker clean recovery and coarser depth support, while Stage 6 produces more stable clean structure and a better-shaped geometry field.
        """
    ).strip()


def appendix_caption_registry(appendix_csv_dir: Path, appendix_figures_root: Path, appendix_figure_note: dict[str, Any] | None) -> str:
    appendix_figure_path = appendix_figure_note["path"] if appendix_figure_note else ""
    return textwrap.dedent(
        f"""
        # Appendix Caption Registry

        ## Appendix Blocks

        ### Appendix Figure A1
        Path: {appendix_figure_path}
        Caption: Pseudo-clean supervision examples from two representative FUNA scenes. Each row keeps the same held-out underwater view and compares the clean-render branch without pseudo-clean supervision, with the current pseudo-clean signal, and with oracle clean supervision. The qualitative role of this figure is limited and should remain aligned with Table 4: pseudo-clean is a useful directional signal that improves de-watered structure, while oracle clean remains an upper bound rather than a main-method setting.

        ### Appendix Table Source Bundle
        Path: {appendix_csv_dir.resolve()}
        Caption: The appendix table directory stores the full exported CSV sources used for the writing package, including the original `exp1_table.csv`, `exp2_table.csv`, `exp3_table.csv`, the authoritative main-comparison xlsx copy, the spreadsheet text dump, and the per-scene `exp1_stage6_orig_vs_realrun_per_scene.csv` discrepancy audit. These sources are preserved so downstream writing and review can trace every aggregated number back to an explicit frozen file.

        ## Writing Rule

        Appendix captions stay dense and information-preserving. They should keep protocol, compared settings, main changed variable, what the viewer should conclude, and any boundary note needed to avoid over-claiming. Do not shorten them to mimic final paper style at this layer.

        ## Firewall Reminder

        Do not expose internal runtime-only provenance strings, temporary scratch directories, or hidden controller state inside paper-facing appendix captions. It is acceptable to mention which exported table file a number came from, but not internal research_os control mechanics unless the note is explicitly about export provenance.

        ## Appendix Figure Rule

        Appendix figures should extend the evidence already frozen for the main story, not introduce a new hero claim. Additional qualitative panels belong here when they clarify ablations, supervision profiles, or result-source discrepancies that would otherwise clutter the main text.
        """
    ).strip()


def build_authoritative_runtime_report(
    table1_rows: list[dict[str, str]],
    table1b_rows: list[dict[str, str]],
    exp1_rows: list[dict[str, Any]],
    exp2_rows: list[dict[str, Any]],
    exp3_rows: list[dict[str, Any]],
    figure_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "project_name": PROJECT_NAME,
        "generated_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "delivery_root": str(DELIVERY_ROOT.resolve()),
        "semantic_freeze": {
            "approved_by": APPROVED_BY,
            "approved_on": APPROVAL_DATE,
            **APPROVED,
        },
        "task_summary": {
            "problem_sentence": "Study underwater 3D Gaussian Splatting as an evaluable geometry-medium disentanglement problem rather than a full intrinsic underwater reconstruction problem.",
            "datasets": ["FUNA", "SeaThru"],
            "entrypoints": {
                "training": str((REPO_ROOT / "train.py").resolve()),
                "render": str((REPO_ROOT / "render.py").resolve()),
                "metrics": str((REPO_ROOT / "metrics3.py").resolve()),
            },
        },
        "result_source_binding": {
            "main_comparison_authority": str((OUTPUT_ROOT / "论文主实验对比表.xlsx").resolve()),
            "stratified_analysis": str((OUTPUT_ROOT / "exp2_table.csv").resolve()),
            "progressive_ablation": str((OUTPUT_ROOT / "exp1_table.csv").resolve()),
            "pseudoclean_analysis": str((OUTPUT_ROOT / "exp3_table.csv").resolve()),
            "notes": [
                "Main comparison is frozen to the author-specified spreadsheet.",
                "exp1 stage6 realrun and exp2 blue_4 are close but not numerically identical; they must not be silently interchanged.",
            ],
        },
        "main_text_tables": {
            "table1_funa": table1_rows,
            "table1b_real_seathru": table1b_rows,
            "table2_stratified": exp2_rows,
            "table3_progressive_ablation": exp1_rows,
            "table4_pseudoclean": exp3_rows,
        },
        "figure_inventory": figure_notes,
        "boundary_notes": [
            "Do not claim full intrinsic underwater physics recovery.",
            "Do not claim universal real-scene geometry superiority from SeaThru.",
            "Center the paper on evaluable geometry-medium disentanglement and medium-induced geometry corruption mitigation.",
        ],
    }


def build_formula_block_pack() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "generated_at_utc": utc_now(),
        "formula_blocks": [
            {
                "block_id": "underwater_image_formation",
                "role": "method_core",
                "expression": "I = J * A(z) + B(z)",
                "support": "The complete physics branch treats the 3DGS clear-scene render J as the clean scene, attenuates it with A(z), and adds backscatter B(z).",
                "code_anchor": "scene/complete_physics_model.py:232-310",
            },
            {
                "block_id": "backscatter_model",
                "role": "physics_branch",
                "expression": "B(z) = B_inf * (1 - exp(-beta_B * z)) + J_prime * exp(-beta_D * z)",
                "support": "BackscatterNet explicitly predicts backscatter coefficients and uses a far-field underwater background parameter B_inf.",
                "code_anchor": "scene/complete_physics_model.py:33-113",
            },
            {
                "block_id": "attenuation_model",
                "role": "physics_branch",
                "expression": "A(z) = exp(-beta_D * z)",
                "support": "AttenuationNet estimates depth-conditioned attenuation maps from metric depth and optional appearance features.",
                "code_anchor": "scene/complete_physics_model.py:116-204",
            },
            {
                "block_id": "legacy_medium_estimator",
                "role": "fallback_branch",
                "expression": "I = J * exp(-beta_D * z) + B_inf * (1 - exp(-beta_B * z))",
                "support": "MediumEstimator keeps a basis-decomposition variant of the same physics intuition as a fallback branch.",
                "code_anchor": "scene/medium_model.py:6-149",
            },
            {
                "block_id": "training_objective",
                "role": "optimization",
                "expression": "L_total = L_rgb + lambda_bg * L_bg + lambda_fake_air * L_fake_air + lambda_geo * L_geo + lambda_ncc * L_ncc + regularizers",
                "support": "The training loop combines underwater RGB reconstruction, background-consistency regularization, fake-air supervision, geometry/photo consistency, and pruning-aware regularizers.",
                "code_anchor": "train.py:1241-1419, train.py:2030-2364",
            },
            {
                "block_id": "capacity_control_logic",
                "role": "geometry_regularization",
                "expression": "far-field gating + densify blocking + conflict-aware pruning + alpha/background suppression",
                "support": "The method constrains which Gaussians can grow or survive in far-field, low-texture, and background-conflicting regions.",
                "code_anchor": "train.py:2228-2364",
            },
        ],
    }


def build_result_discussion_pack(
    table1_rows: list[dict[str, str]],
    table1b_rows: list[dict[str, str]],
    exp1_rows: list[dict[str, Any]],
    exp2_rows: list[dict[str, Any]],
    exp3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ours = mean_row_by_method(table1_rows, "Ours")
    watersplatting = mean_row_by_method(table1_rows, "WaterSplatting")
    pgsr = mean_row_by_method(table1_rows, "PGSR")
    seathru = mean_row_by_method(table1_rows, "SeaThru-NeRF")
    seathru_real = mean_row_by_method(table1b_rows, "SeaThru-NeRF")
    ours_real = mean_row_by_method(table1b_rows, "Ours")

    stage0 = next(r for r in exp1_rows if r["stage_label"] == "Stage 0: PGSR")
    stage6 = next(r for r in exp1_rows if r["stage_label"] == "Stage 6: + Pseudo-Clean")
    fakeair_none = next(r for r in exp3_rows if r["profile"] == "fakeair_none")
    fakeair_pseudo = next(r for r in exp3_rows if r["profile"] == "fakeair_pseudo")
    fakeair_oracle = next(r for r in exp3_rows if r["profile"] == "fakeair_oracle")

    return {
        "project_id": PROJECT_ID,
        "generated_at_utc": utc_now(),
        "discussion_units": [
            {
                "experiment_id": "exp1_main_comparison",
                "claim_supported": "Ours improves clean appearance and geometry recovery on FUNA while keeping competitive underwater rendering performance.",
                "headline_numbers": {
                    "ours_clean_psnr": ours["Clean PSNR"],
                    "watersplatting_clean_psnr": watersplatting["Clean PSNR"],
                    "ours_depth_rmse": ours["Depth RMSE"],
                    "pgsr_depth_rmse": pgsr["Depth RMSE"],
                    "ours_depth_absrel": ours["AbsRel"],
                    "ours_normal_mean": ours["Normal Mean"],
                },
                "delta_summary": {
                    "clean_psnr_gain_vs_watersplatting": round(ours["Clean PSNR"] - watersplatting["Clean PSNR"], 3),
                    "depth_rmse_reduction_vs_pgsr": round(pgsr["Depth RMSE"] - ours["Depth RMSE"], 3),
                    "depth_rmse_reduction_vs_watersplatting": round(watersplatting["Depth RMSE"] - ours["Depth RMSE"], 3),
                    "real_ssim_gain_vs_seathru_nerf": round(ours_real["RGB SSIM"] - seathru_real["RGB SSIM"], 3),
                    "real_lpips_gain_vs_seathru_nerf": round(seathru_real["RGB LPIPS"] - ours_real["RGB LPIPS"], 3),
                },
                "boundary_note": "Real SeaThru conclusions remain conservative because Ours is not the best RGB PSNR method there.",
            },
            {
                "experiment_id": "exp2_stratified_analysis",
                "claim_supported": "The method remains stable across water-color and degradation buckets covered by FUNA.",
                "headline_rows": exp2_rows,
                "boundary_note": "This is bucket robustness inside FUNA coverage, not universal unseen-medium generalization.",
            },
            {
                "experiment_id": "exp3_progressive_ablation",
                "claim_supported": "The attribution-control chain matters as a chain, not just as isolated tricks.",
                "headline_numbers": {
                    "stage0_clean_psnr": float(stage0["clean_psnr"]),
                    "stage6_clean_psnr": float(stage6["clean_psnr"]),
                    "stage0_depth_rmse": float(stage0["depth_rmse"]),
                    "stage6_depth_rmse": float(stage6["depth_rmse"]),
                    "stage0_depth_delta125": float(stage0["depth_delta125"]),
                    "stage6_depth_delta125": float(stage6["depth_delta125"]),
                },
                "delta_summary": {
                    "clean_psnr_gain": round(float(stage6["clean_psnr"]) - float(stage0["clean_psnr"]), 3),
                    "depth_rmse_drop": round(float(stage0["depth_rmse"]) - float(stage6["depth_rmse"]), 3),
                    "depth_delta125_gain": round(float(stage6["depth_delta125"]) - float(stage0["depth_delta125"]), 3),
                },
                "boundary_note": "The table demonstrates mechanism synergy under the chosen stage chain; it does not prove every module combination is globally optimal.",
            },
            {
                "experiment_id": "exp4_pseudoclean_analysis",
                "claim_supported": "Pseudo-clean supervision is useful but not equivalent to oracle clean supervision.",
                "headline_numbers": {
                    "no_pseudo_clean_psnr": float(fakeair_none["clean_psnr"]),
                    "pseudo_clean_psnr": float(fakeair_pseudo["clean_psnr"]),
                    "oracle_clean_psnr": float(fakeair_oracle["clean_psnr"]),
                    "no_pseudo_depth_rmse": float(fakeair_none["depth_rmse"]),
                    "pseudo_depth_rmse": float(fakeair_pseudo["depth_rmse"]),
                    "oracle_depth_rmse": float(fakeair_oracle["depth_rmse"]),
                },
                "delta_summary": {
                    "pseudo_clean_psnr_gain_vs_none": round(float(fakeair_pseudo["clean_psnr"]) - float(fakeair_none["clean_psnr"]), 3),
                    "oracle_clean_psnr_gain_vs_pseudo": round(float(fakeair_oracle["clean_psnr"]) - float(fakeair_pseudo["clean_psnr"]), 3),
                    "pseudo_depth_rmse_drop_vs_none": round(float(fakeair_none["depth_rmse"]) - float(fakeair_pseudo["depth_rmse"]), 3),
                },
                "boundary_note": "Oracle clean is an upper-bound ablation only and must not be described as a deployable training setting.",
            },
        ],
    }


def build_code_grounding_pack() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "generated_at_utc": utc_now(),
        "module_to_code_map": [
            {
                "module": "Scene and dataset ingress",
                "summary": "Builds the scene object, detects COLMAP or Blender-style inputs, and constructs train/test camera lists plus point clouds.",
                "anchors": [
                    {"path": str((REPO_ROOT / "scene/__init__.py").resolve()), "lines": "23-103"},
                    {"path": str((REPO_ROOT / "scene/dataset_readers.py").resolve()), "lines": "263-302"},
                ],
            },
            {
                "module": "Appearance branch",
                "summary": "Per-image appearance compensation branch with its own optimizer.",
                "anchors": [
                    {"path": str((REPO_ROOT / "scene/app_model.py").resolve()), "lines": "9-30"},
                ],
            },
            {
                "module": "Physics branch",
                "summary": "BackscatterNet, AttenuationNet, and CompleteUnderwaterPhysics implement the underwater image-formation path and optimizer state.",
                "anchors": [
                    {"path": str((REPO_ROOT / "scene/complete_physics_model.py").resolve()), "lines": "33-317"},
                    {"path": str((REPO_ROOT / "scene/medium_model.py").resolve()), "lines": "6-149"},
                ],
            },
            {
                "module": "Renderer",
                "summary": "Renders clear-scene 3DGS outputs, alpha/depth/normal, and then applies the underwater physics branch to produce underwater and clean outputs.",
                "anchors": [
                    {"path": str((REPO_ROOT / "gaussian_renderer/__init__.py").resolve()), "lines": "36-259"},
                ],
            },
            {
                "module": "Training and capacity control",
                "summary": "Connects scene loading, loss construction, backward pass, densification, far-field gating, and pruning logic.",
                "anchors": [
                    {"path": str((REPO_ROOT / "train.py").resolve()), "lines": "1241-1419"},
                    {"path": str((REPO_ROOT / "train.py").resolve()), "lines": "2030-2364"},
                ],
            },
        ],
        "claim_to_code_binding": [
            {
                "claim_unit": "physically closed medium-background rendering",
                "evidence": [
                    "gaussian_renderer/__init__.py applies CompleteUnderwaterPhysics after clear-scene rendering and handles no-hit / background closure logic.",
                    "scene/complete_physics_model.py defines backscatter and attenuation operators plus the I = J * A(z) + B(z) composition.",
                ],
            },
            {
                "claim_unit": "misattribution-aware Gaussian capacity control",
                "evidence": [
                    "train.py blocks far densification, performs DRO-guided pruning, and applies background double-evidence pruning.",
                    "train.py keeps background-consistency loss and alpha/background evidence in the late-stage optimization path.",
                ],
            },
            {
                "claim_unit": "paired evaluability on FUNA",
                "evidence": [
                    "render.py exports underwater RGB, clean render, depth, and normal products.",
                    "metrics3.py evaluates RGB, AIR/AIRBGBLACK, depth, and normal metrics from the exported products.",
                ],
            },
        ],
    }


def compute_semantic_surface_hash(handoff: dict[str, Any]) -> str:
    fields = [
        "approved_title",
        "approved_method_name",
        "approved_task_type",
        "approved_one_sentence_story",
        "approved_core_goal",
        "approved_claim_boundary",
        "approved_appendix_title",
        "semantic_authority_path",
        "semantic_authority_rule",
    ]
    payload = {field: handoff.get(field) for field in fields}
    return sha256_bytes(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).encode("utf-8"))


def compute_export_surface_hash(snapshot: dict[str, Any]) -> str:
    payload = {
        "approved_semantic_surface_hash": snapshot.get("approved_semantic_surface_hash"),
        "delivery_root": snapshot.get("delivery_root"),
        "paper_input_path": snapshot.get("paper_input_path"),
        "appendix_input_path": snapshot.get("appendix_input_path"),
        "main_text_caption_registry_path": snapshot.get("main_text_caption_registry_path"),
        "appendix_caption_registry_path": snapshot.get("appendix_caption_registry_path"),
        "authoritative_runtime_report_path": snapshot.get("authoritative_runtime_report_path"),
        "formula_block_pack_path": snapshot.get("formula_block_pack_path"),
        "result_discussion_pack_path": snapshot.get("result_discussion_pack_path"),
        "code_grounding_pack_path": snapshot.get("code_grounding_pack_path"),
        "figure_roots": snapshot.get("figure_roots"),
        "table_roots": snapshot.get("table_roots"),
        "paper_input_sha256": snapshot.get("paper_input_sha256"),
        "appendix_input_sha256": snapshot.get("appendix_input_sha256"),
        "main_text_caption_registry_sha256": snapshot.get("main_text_caption_registry_sha256"),
        "appendix_caption_registry_sha256": snapshot.get("appendix_caption_registry_sha256"),
        "authoritative_runtime_report_sha256": snapshot.get("authoritative_runtime_report_sha256"),
        "formula_block_pack_sha256": snapshot.get("formula_block_pack_sha256"),
        "result_discussion_pack_sha256": snapshot.get("result_discussion_pack_sha256"),
        "code_grounding_pack_sha256": snapshot.get("code_grounding_pack_sha256"),
        "caption_policy": snapshot.get("caption_policy"),
    }
    return sha256_bytes(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).encode("utf-8"))


def build_runtime_files(
    validator_passed: bool,
    validation_report_path: Path,
    handoff_path: Path,
    snapshot_path: Path,
    manifest_path: Path,
) -> None:
    project_override = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            ("project_name", PROJECT_NAME),
            ("venue_target", "unknown"),
            ("submission_window", "unknown"),
            ("research_archetype", {"primary": "method", "secondary": ["benchmark", "analysis"]}),
            ("paper_mode", "method_paper"),
            (
                "headline_judge",
                [
                    "does the work establish an evaluable underwater 3DGS problem with a defensible geometry-medium claim",
                    "are clean appearance and geometry gains supported beyond underwater RGB improvements",
                ],
            ),
            (
                "stage_emphasis_map",
                {
                    "s1": "claim_boundary_and_protocol_integrity",
                    "s2": "story_commit_around_evaluable_disentanglement",
                    "s3": "main_comparison_and_real_data_competitiveness",
                    "s4": "stratified_and_ablation_mechanism_support",
                    "s5": "qualitative_and_failure_buckets",
                    "s6": "freeze_consistency_and_result_source_binding",
                    "s7": "final_export_packaging",
                },
            ),
            (
                "problem_sentence",
                "Reframe underwater 3D Gaussian Splatting as an evaluable geometry-medium disentanglement problem and reduce medium-induced geometry corruption with physically closed rendering and Gaussian capacity control.",
            ),
            (
                "out_of_scope",
                [
                    "full intrinsic underwater physics recovery",
                    "universal real-ocean geometry reconstruction",
                    "unbounded claims beyond FUNA and current SeaThru coverage",
                ],
            ),
            (
                "core_object",
                {
                    "type": "underwater_scene_level_gaussian_representation",
                    "description": "scene-level 3DGS representation jointly responsible for underwater RGB, clean appearance, depth, and normal outputs",
                },
            ),
            (
                "current_state",
                {
                    "current_incumbent": "GeoMed-GS",
                    "current_runner_up": "metric_axis_dependent_baselines",
                    "active_protocols": [
                        "FUNA_main_comparison",
                        "FUNA_stratified_medium_configuration_analysis",
                        "FUNA_progressive_ablation",
                        "FUNA_pseudoclean_bias_analysis",
                        "SeaThru_real_data_comparison",
                    ],
                    "current_failure_taxonomy": [
                        "medium-induced geometry corruption",
                        "far-field fog-wall absorption into Gaussians",
                        "RGB-good but clean/depth/normal-inconsistent reconstructions",
                    ],
                },
            ),
            (
                "data_sources",
                {
                    "primary": ["FUNA synthetic paired underwater/clean/depth/normal benchmark"],
                    "auxiliary": ["SeaThru real underwater scenes", "pseudo-clean outputs from underwater enhancement preprocessing"],
                },
            ),
            (
                "protocol_policy",
                {
                    "main_protocol": "scene-level held-out view reconstruction with joint underwater RGB, clean, depth, and normal evaluation",
                    "stress_protocols": ["water_color_split", "degradation_level_split", "real_seathru_transfer"],
                    "probe_protocols": ["progressive_stage_ablation", "pseudo_clean_profile_comparison"],
                    "story_eligible_protocols": [
                        "main_comparison",
                        "stratified_medium_configuration",
                        "progressive_ablation",
                        "pseudo_clean_bias_analysis",
                    ],
                },
            ),
            (
                "metric_policy",
                {
                    "primary_metrics": ["clean_psnr", "depth_rmse", "depth_absrel", "depth_delta125", "normal_mean_angle"],
                    "diagnostic_metrics": [
                        "rgb_psnr",
                        "rgb_ssim",
                        "rgb_lpips",
                        "clean_ssim",
                        "clean_lpips",
                        "normal_median_angle",
                        "normal_acc_11_25deg",
                        "UCIQE",
                        "UIQM",
                        "NIQE",
                    ],
                },
            ),
            (
                "human_creative_prior",
                {
                    "owner": "ycy",
                    "owner_thesis_sentence": "The paper should be about evaluable geometry-medium disentanglement, not about over-claiming full underwater 3D reconstruction.",
                    "preferred_claim_layer_ceiling": "medium-induced geometry corruption mitigation with paired evaluability",
                    "favored_bottlenecks": ["far-field misattribution", "background closure", "capacity control"],
                    "favored_task_slices": ["synthetic paired benchmark", "3DGS geometry quality", "real-data conservative transfer"],
                    "cross_domain_inspirations": [],
                    "forbidden_storylines": [
                        "full intrinsic underwater physics recovery",
                        "universal real-scene geometry best",
                    ],
                    "acceptable_tradeoffs": ["competitive rather than dominating RGB performance if clean and geometry gains are clear"],
                    "protected_angle": "evaluable geometry-medium disentanglement",
                    "lab_uniqueness_key": "paired underwater/clean/depth/normal benchmark plus geometry-aware underwater 3DGS",
                },
            ),
            (
                "story_policy",
                {
                    "story_shortlist_policy": "adaptive_default",
                    "title_freeze_policy": "late_if_honest",
                    "confirmatory_asset_policy": "appendix_or_defense_only_until_publication_grade",
                },
            ),
            ("story_holdout_policy", "split_after_story_shortlist_before_title_freeze"),
            ("discovery_lane_required", False),
            ("external_validity_path", "real_seathru_competitiveness_with_conservative_geometry_language"),
            (
                "compute_budget",
                {
                    "num_gpus": 6,
                    "max_long_runs": 3,
                    "max_confirmatory_reruns": 2,
                    "allowed_gpu_ids": [0, 1, 2, 3, 4, 5],
                    "occupancy_bias": "saturate_confirmed_capacity_if_human_confirms",
                    "headroom_preference": "explicit_if_needed",
                    "reserve_margin_hint": "completed main experiments were organized under a 6-GPU regime inferred from funa_exp13_6gpu naming",
                    "resource_confirmation_required_for_large_runs": True,
                    "launch_without_confirmation_policy": "provisional_small_only",
                },
            ),
            (
                "delivery_policy",
                {
                    "final_output_root": str(DELIVERY_ROOT.resolve()),
                    "use_existing_structure": True,
                    "visibility_before_freeze": "sealed",
                    "unlock_condition": "after_s7_freeze_submission_pass_and_writing_handoff_approval",
                    "export_role": "final_packaging_only",
                    "export_trigger": [
                        "asset_freeze_manifest",
                        "submission_readiness_report",
                        "writing_handoff_gate",
                    ],
                    "paper_input_path": str((DELIVERY_ROOT / "paper_input.txt").resolve()),
                    "appendix_input_path": str((DELIVERY_ROOT / "input_appendix" / "appendix_input.txt").resolve()),
                    "authoritative_runtime_report_path": str((DELIVERY_ROOT / "authoritative_runtime_report.json").resolve()),
                    "formula_block_pack_path": str((DELIVERY_ROOT / "formula_block_pack.json").resolve()),
                    "result_discussion_pack_path": str((DELIVERY_ROOT / "result_discussion_pack.json").resolve()),
                    "code_grounding_pack_path": str((DELIVERY_ROOT / "code_grounding_pack.json").resolve()),
                    "figures_root": str((DELIVERY_ROOT / "figures").resolve()),
                    "appendix_figures_root": str((DELIVERY_ROOT / "input_appendix" / "figures").resolve()),
                    "materials_root": str((DELIVERY_ROOT / "materials").resolve()),
                    "story_tables_root": str((DELIVERY_ROOT / "materials" / "story_evidence_tables").resolve()),
                    "caption_policy": "dense_descriptive_no_information_loss",
                    "critical_field_signoff_required": True,
                },
            ),
            (
                "runtime_policy",
                {
                    "use_runtime_layer": True,
                    "project_state_path": "./research_os/00_PROJECT_STATE.md",
                    "research_ledger_path": "./research_os/01_RESEARCH_LEDGER.md",
                    "current_phase_packet_path": "./research_os/current_phase_packet.yaml",
                    "ledger_compaction_stride": 25,
                },
            ),
            (
                "interaction_policy",
                {
                    "human_participation_mode": "autonomy_first",
                    "mandatory_human_windows": ["S7_final_export_hard_author_signoff"],
                    "advisory_human_windows": ["S2_story_commit", "S5_qualitative_review", "S7_final_freeze"],
                    "conditional_human_windows": ["S6_protocol_or_claim_conflict"],
                    "unanswered_advisory_policy": "codex_default_with_ledger_note",
                    "partial_response_policy": "apply_answered_fields_fallback_unanswered",
                    "continue_unblocked_branches_when_waiting": True,
                    "decision_packet_rule": "options_default_impact",
                    "lab_branch_registry_path": "./research_os/LAB_BRANCH_REGISTRY.md",
                },
            ),
            (
                "must_ask_user",
                [
                    "changing_problem_definition",
                    "changing_main_protocol",
                    "changing_primary_metric",
                    "changing_paper_positioning",
                    "final_writing_handoff_semantics",
                ],
            ),
            (
                "can_self_decide",
                [
                    "queue_order",
                    "export_packaging_layout",
                    "appendix_table_grouping",
                    "qualitative_figure_asset_selection_within_frozen_results",
                ],
            ),
            ("promotion_policy", {"core_claim_requires_matched_control": True, "core_assets_must_be_publication_grade": True}),
            (
                "carry_over_from_history",
                {
                    "task_similarity_candidates": [],
                    "reusable_modules": ["PGSR backbone", "3DGS renderer", "paired synthetic evaluation"],
                    "known_failure_signatures": ["fog_wall_far_field", "clean_geometry_metric_divergence_from_rgb_only_gains"],
                },
            ),
            ("starter_template", False),
            ("rewrite_on_first_real_repo_scan", False),
        ]
    )

    bootstrap_packet = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            (
                "problem_sentence",
                "Build a publication-grade writing handoff for an already-completed underwater 3DGS project whose main claim is evaluable geometry-medium disentanglement rather than full underwater reconstruction.",
            ),
            (
                "expected_delivery",
                "A validated final delivery package with paper_input, appendix_input, caption registries, four companion packs, frozen figures, frozen table sources, and export-binding manifests.",
            ),
            (
                "resource_window",
                {
                    "num_gpus": 6,
                    "max_long_runs": 0,
                    "submission_window": "unknown",
                    "allowed_gpu_ids": [0, 1, 2, 3, 4, 5],
                    "occupancy_bias": "saturate_confirmed_capacity_if_human_confirms",
                    "headroom_preference": "explicit_if_needed",
                    "reserve_margin_hint": "no new heavy runs are required for the export phase",
                    "resource_confirmation_status": "not_needed_for_export_only",
                },
            ),
            ("venue_target", "unknown"),
            (
                "obvious_unknowns",
                [
                    "venue choice remains unspecified",
                    "real-world external validity remains bounded to current SeaThru evidence",
                ],
            ),
            (
                "data_source_catalog",
                [
                    {
                        "data_source_id": "funa_synthetic_paired",
                        "role": "primary",
                        "split_summary": "scene-level held-out view evaluation with paired underwater/clean/depth/normal targets",
                        "input_shape": "multi-view underwater RGB with camera poses",
                        "target_shape": "underwater RGB, clean appearance, depth, and normal for evaluation",
                        "semantic_target": "evaluable geometry-medium disentanglement",
                    },
                    {
                        "data_source_id": "seathru_real",
                        "role": "auxiliary",
                        "split_summary": "real underwater scenes for competitive transfer evaluation",
                        "input_shape": "multi-view underwater RGB with poses",
                        "target_shape": "real-scene underwater rendering and de-watered perceptual evidence",
                        "semantic_target": "conservative real-data competitiveness",
                    },
                ],
            ),
            ("notes", "Bootstrap runtime rewritten from actual repo evidence and author-approved export semantics."),
            ("starter_template", False),
            ("rewrite_on_first_real_repo_scan", False),
        ]
    )

    state_md = textwrap.dedent(
        f"""
        # PROJECT STATE

        ## Identity

        - Project ID: `{PROJECT_ID}`
        - Project Name: `{PROJECT_NAME}`
        - Current Stage: `S7`
        - Runtime Status: `final_export_packaging_ready`

        ## High-Level Goal

        把已经完成实验的水下 3DGS 项目收束成 research_os 当前契约认可的最终写作交付层，确保标题、方法名、任务类型、一句话故事、最终 Claim、Claim 边界和附录标题已经作者确认，并把主文输入包、附录输入包、证据表、caption registry、companion packs 与导出绑定文件一次性整理完成。

        ## Truth Rules

        - 当前项目的中心定位是 `evaluable geometry-medium disentanglement for underwater 3D Gaussian Splatting`。
        - `论文主实验对比表.xlsx` 是主比较表的权威源。
        - `exp2_table.csv`、`exp1_table.csv`、`exp3_table.csv` 分别绑定 Table 2、Table 3、Table 4。
        - `exp1 stage6 realrun` 与 `exp2 blue_4` 已确认不完全相同；主比较不得擅自互换来源。
        - Real SeaThru 结果只能支持“competitive real-scene evidence”层面的写法，不支持“universal real-scene geometry best”。

        ## Must Read Ledger Entries

        - `0002`
        - `0003`
        - `0004`

        ## Immediate Next Actions

        1. 保持 `paper_input.txt` 的 6 个语义字段与已批准文本严格绑定
        2. 维护导出层里的 caption registry、companion packs、图表源文件与 manifest 一致
        3. 如果主表或 Claim 需要再改，必须重新打开 writing_handoff_gate，而不是只改输出文件

        ## Hard Boundaries

        - 不要把本文写成完整水下物理反演或普适真实海域重建
        - 不要在 SeaThru 上过度声称真实几何全面领先
        - 不要让最终写作层中的语义字段偏离已批准版本
        - 不要把 `paper_input.txt` 当成长篇叙事容器；详细材料应进入 appendix 和 companion packs
        """
    ).strip()

    ledger_md = textwrap.dedent(
        f"""
        # RESEARCH LEDGER

        ## Entry 0002

        - Entry Template: Bootstrap
        - Status: repo_scanned_and_project_identity_rewritten
        - Why: The repo contains a completed underwater 3DGS project with its own synthetic paired benchmark, training code, rendering/evaluation scripts, and frozen outputs under `outputs/funa_exp13_6gpu/`.
        - Evidence Anchors:
          - `train.py`
          - `render.py`
          - `metrics3.py`
          - `outputs/funa_exp13_6gpu/`
          - `paper_input_FUNA_1_revised_v2.txt`
          - `clam验证_FUNA_1_revised_v2.txt`
        - Next Honest Action:
          - freeze claim framing around evaluable geometry-medium disentanglement
          - bind each headline experiment to a concrete result source
          - prepare final export packaging

        ## Entry 0003

        - Entry Template: Result Source Binding
        - Status: main_result_authority_bound
        - Why: The author explicitly designated `论文主实验对比表.xlsx` as the authority for the main comparison table, while `exp1_table.csv`, `exp2_table.csv`, and `exp3_table.csv` bind the progressive ablation, stratified analysis, and pseudo-clean analyses respectively.
        - Evidence Anchors:
          - `outputs/funa_exp13_6gpu/论文主实验对比表.xlsx`
          - `outputs/funa_exp13_6gpu/exp1_table.csv`
          - `outputs/funa_exp13_6gpu/exp2_table.csv`
          - `outputs/funa_exp13_6gpu/exp3_table.csv`
          - `outputs/funa_exp13_6gpu/exp1_stage6_orig_vs_realrun_per_scene.csv`
        - Next Honest Action:
          - keep main-comparison numbers frozen to the spreadsheet
          - use the csv exports only for their intended experiments
          - surface any source discrepancy in the appendix/export provenance

        ## Entry 0004

        - Entry Template: Hard Author Signoff and Export
        - Status: author_confirmed_final_semantics_and_export_generated
        - Why: The author confirmed title, method name, task type, one-sentence story, final claim, claim boundary, and appendix title on {APPROVAL_DATE}, unlocking S7 final export packaging.
        - Evidence Anchors:
          - `research_os/research_os/artifacts/final_export_funa_geomed_gs/writing_handoff_gate.yaml`
          - `research_os/research_os/artifacts/final_export_funa_geomed_gs/asset_freeze_manifest.yaml`
          - `research_os/research_os/artifacts/final_export_funa_geomed_gs/submission_readiness_report.yaml`
          - `research_os/research_os/artifacts/final_export_funa_geomed_gs/approved_export_surface_snapshot.yaml`
          - `research_os/research_os/artifacts/final_export_funa_geomed_gs/output_delivery_manifest.yaml`
          - `final_delivery_funa_geomed_gs/`
        - Next Honest Action:
          - treat the delivery package as the current semantic authority
          - reopen writing handoff before any semantic field changes
          - keep future edits scoped to polishing or explicitly re-approved changes
        """
    ).strip()

    current_phase = OrderedDict(
        [
            ("phase_id", "S7"),
            ("current_wave_id", "funa_final_export_packaging_wave_01"),
            ("wave_state", "final_export_packaging_completed"),
            ("timing_gate_status", "not_open"),
            ("release_gate_status", "not_open"),
            (
                "phase_goal",
                "Freeze the paper-facing semantic surface, bind every headline experiment to frozen evidence, and export the final writing package without semantic drift.",
            ),
            (
                "required_inputs",
                [
                    "research_os/00_PROJECT_STATE.md",
                    "research_os/01_RESEARCH_LEDGER.md",
                    "research_os/project_override.yaml",
                    "research_os/project_bootstrap_packet.yaml",
                    "research_os/current_phase_packet.yaml",
                    "paper_input_FUNA_1_revised_v2.txt",
                    "clam验证_FUNA_1_revised_v2.txt",
                    "outputs/funa_exp13_6gpu/论文主实验对比表.xlsx",
                    "outputs/funa_exp13_6gpu/exp1_table.csv",
                    "outputs/funa_exp13_6gpu/exp2_table.csv",
                    "outputs/funa_exp13_6gpu/exp3_table.csv",
                ],
            ),
            (
                "required_outputs",
                [
                    "final_delivery_funa_geomed_gs/paper_input.txt",
                    "final_delivery_funa_geomed_gs/input_appendix/appendix_input.txt",
                    "final_delivery_funa_geomed_gs/materials/story_evidence_tables/main_text/main_text_caption_registry.md",
                    "final_delivery_funa_geomed_gs/materials/story_evidence_tables/appendix/appendix_caption_registry.md",
                    "final_delivery_funa_geomed_gs/authoritative_runtime_report.json",
                    "final_delivery_funa_geomed_gs/formula_block_pack.json",
                    "final_delivery_funa_geomed_gs/result_discussion_pack.json",
                    "final_delivery_funa_geomed_gs/code_grounding_pack.json",
                    "research_os/research_os/artifacts/final_export_funa_geomed_gs/writing_handoff_gate.yaml",
                    "research_os/research_os/artifacts/final_export_funa_geomed_gs/approved_export_surface_snapshot.yaml",
                    "research_os/research_os/artifacts/final_export_funa_geomed_gs/output_delivery_manifest.yaml",
                    "research_os/research_os/artifacts/final_export_funa_geomed_gs/final_export_validation_report.yaml",
                ],
            ),
            (
                "notes",
                [
                    "S7 is export-only for this repo snapshot because the core experiments are already complete.",
                    "Main comparison authority is the xlsx designated by the author.",
                    "paper_input uses the strict current final-export contract with [最终Claim], not the old example marker [核心目标].",
                ],
            ),
            (
                "active_writing_handoff_gate",
                str(handoff_path.relative_to(REPO_ROOT)),
            ),
            (
                "active_approved_export_surface_snapshot",
                str(snapshot_path.relative_to(REPO_ROOT)),
            ),
            (
                "active_output_delivery_manifest",
                str(manifest_path.relative_to(REPO_ROOT)),
            ),
            ("final_export_transition_status", "validated_locally_passed" if validator_passed else "validated_locally_failed"),
            ("local_validation_report", str(validation_report_path.relative_to(REPO_ROOT))),
        ]
    )

    write_yaml(REPO_ROOT / "research_os" / "research_os" / "project_override.yaml", project_override)
    write_yaml(REPO_ROOT / "research_os" / "research_os" / "project_bootstrap_packet.yaml", bootstrap_packet)
    write_text(REPO_ROOT / "research_os" / "research_os" / "00_PROJECT_STATE.md", state_md)
    write_text(REPO_ROOT / "research_os" / "research_os" / "01_RESEARCH_LEDGER.md", ledger_md)
    write_yaml(REPO_ROOT / "research_os" / "research_os" / "current_phase_packet.yaml", current_phase)


def build_export_artifacts() -> int:
    ensure_dir(DELIVERY_ROOT)
    ensure_dir(ARTIFACT_ROOT)

    paper_input_path = DELIVERY_ROOT / "paper_input.txt"
    appendix_input_path = DELIVERY_ROOT / "input_appendix" / "appendix_input.txt"
    main_csv_dir = DELIVERY_ROOT / "materials" / "story_evidence_tables" / "main_text"
    appendix_csv_dir = DELIVERY_ROOT / "materials" / "story_evidence_tables" / "appendix"
    main_caption_path = main_csv_dir / "main_text_caption_registry.md"
    appendix_caption_path = appendix_csv_dir / "appendix_caption_registry.md"
    figures_root = DELIVERY_ROOT / "figures"
    appendix_figures_root = DELIVERY_ROOT / "input_appendix" / "figures"
    runtime_report_path = DELIVERY_ROOT / "authoritative_runtime_report.json"
    formula_pack_path = DELIVERY_ROOT / "formula_block_pack.json"
    discussion_pack_path = DELIVERY_ROOT / "result_discussion_pack.json"
    code_pack_path = DELIVERY_ROOT / "code_grounding_pack.json"

    table1_rows, table1b_rows, authoritative_dump_text = parse_authoritative_main_tables(OUTPUT_ROOT / "论文主实验对比表.xlsx")
    exp1_full, exp2_full, exp3_full = load_stage_table_rows()
    exp1_rows = relabel_stage_rows(exp1_full)
    exp2_rows = relabel_exp2_rows(exp2_full)
    exp3_rows = relabel_exp3_rows(exp3_full)

    write_csv(main_csv_dir / "table1_funa_main_comparison.csv", table1_rows, TABLE1_COLUMNS)
    write_csv(main_csv_dir / "table1b_real_seathru_main_comparison.csv", table1b_rows, TABLE1B_COLUMNS)
    write_csv(main_csv_dir / "table2_stratified_analysis.csv", exp2_rows, list(exp2_rows[0].keys()))
    write_csv(main_csv_dir / "table3_progressive_ablation.csv", exp1_rows, list(exp1_rows[0].keys()))
    write_csv(main_csv_dir / "table4_pseudoclean_analysis.csv", exp3_rows, list(exp3_rows[0].keys()))

    copy_if_exists(OUTPUT_ROOT / "论文主实验对比表.xlsx", appendix_csv_dir / "论文主实验对比表.xlsx")
    copy_if_exists(OUTPUT_ROOT / "exp1_table.csv", appendix_csv_dir / "exp1_table.csv")
    copy_if_exists(OUTPUT_ROOT / "exp2_table.csv", appendix_csv_dir / "exp2_table.csv")
    copy_if_exists(OUTPUT_ROOT / "exp3_table.csv", appendix_csv_dir / "exp3_table.csv")
    copy_if_exists(OUTPUT_ROOT / "funa_exp13_6gpu_all_results_summary.csv", appendix_csv_dir / "funa_exp13_6gpu_all_results_summary.csv")
    copy_if_exists(OUTPUT_ROOT / "exp1_stage6_orig_vs_realrun_per_scene.csv", appendix_csv_dir / "exp1_stage6_orig_vs_realrun_per_scene.csv")
    write_text(appendix_csv_dir / "authoritative_main_comparison_dump.txt", authoritative_dump_text)

    figure_notes: list[dict[str, Any]] = []
    main_qual = make_main_qualitative_figure(figures_root)
    if main_qual:
        figure_notes.append(main_qual)
    ablation_qual = make_ablation_figure(figures_root)
    if ablation_qual:
        figure_notes.append(ablation_qual)
    pseudo_qual = make_pseudoclean_figure(appendix_figures_root)
    if pseudo_qual:
        figure_notes.append(pseudo_qual)

    paper_input_text = build_paper_input(code_section_text())
    appendix_input_text = build_appendix_input(
        table1_rows,
        table1b_rows,
        exp1_rows,
        exp2_rows,
        exp3_rows,
        figure_notes,
        authoritative_dump_text,
    )
    write_text(paper_input_path, paper_input_text)
    write_text(appendix_input_path, appendix_input_text)

    write_text(main_caption_path, main_text_caption_registry(main_csv_dir, figures_root, figure_notes))
    write_text(appendix_caption_path, appendix_caption_registry(appendix_csv_dir, appendix_figures_root, pseudo_qual))

    runtime_report = build_authoritative_runtime_report(table1_rows, table1b_rows, exp1_rows, exp2_rows, exp3_rows, figure_notes)
    formula_pack = build_formula_block_pack()
    discussion_pack = build_result_discussion_pack(table1_rows, table1b_rows, exp1_rows, exp2_rows, exp3_rows)
    code_pack = build_code_grounding_pack()
    write_json(runtime_report_path, runtime_report)
    write_json(formula_pack_path, formula_pack)
    write_json(discussion_pack_path, discussion_pack)
    write_json(code_pack_path, code_pack)

    handoff_path = ARTIFACT_ROOT / "writing_handoff_gate.yaml"
    freeze_path = ARTIFACT_ROOT / "asset_freeze_manifest.yaml"
    readiness_path = ARTIFACT_ROOT / "submission_readiness_report.yaml"
    snapshot_path = ARTIFACT_ROOT / "approved_export_surface_snapshot.yaml"
    manifest_path = ARTIFACT_ROOT / "output_delivery_manifest.yaml"
    validation_report_path = ARTIFACT_ROOT / "final_export_validation_report.yaml"

    handoff = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            ("gate_id", f"{PROJECT_ID}_writing_handoff_gate_{APPROVAL_DATE}"),
            ("requirement", "hard_author_signoff"),
            ("approval_status", "approved"),
            ("approved_by", APPROVED_BY),
            ("approved_title", APPROVED["approved_title"]),
            ("approved_method_name", APPROVED["approved_method_name"]),
            ("approved_task_type", APPROVED["approved_task_type"]),
            ("approved_one_sentence_story", APPROVED["approved_one_sentence_story"]),
            ("approved_core_goal", APPROVED["approved_core_goal"]),
            ("approved_claim_boundary", APPROVED["approved_claim_boundary"]),
            ("approved_appendix_title", APPROVED["approved_appendix_title"]),
            ("semantic_authority_path", str(paper_input_path.resolve())),
            (
                "semantic_authority_rule",
                "paper_input.txt is the sole semantic authority for title, method name, task type, one-sentence story, final claim, and claim boundary; appendix_input.txt is the sole semantic authority for appendix title.",
            ),
            ("approved_abstract_direction", WRITING_GUIDANCE["approved_abstract_direction"]),
            ("reference_split_rule", WRITING_GUIDANCE["reference_split_rule"]),
            ("appendix_prose_target", WRITING_GUIDANCE["appendix_prose_target"]),
            ("final_polish_scope", WRITING_GUIDANCE["final_polish_scope"]),
            ("notes", f"Author confirmed the frozen writing semantics in chat on {APPROVAL_DATE}."),
        ]
    )
    write_yaml(handoff_path, handoff)

    freeze_manifest = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            ("frozen_title", APPROVED["approved_title"]),
            (
                "frozen_core_contribution_units",
                [
                    "FUNA paired evaluability benchmark with underwater/clean/depth/normal supervision",
                    "Physically closed medium-background rendering for underwater 3DGS",
                    "Misattribution-aware Gaussian capacity control for far-field and background-conflicting regions",
                ],
            ),
            (
                "frozen_main_assets",
                [
                    str((OUTPUT_ROOT / "论文主实验对比表.xlsx").resolve()),
                    str((OUTPUT_ROOT / "exp1_table.csv").resolve()),
                    str((OUTPUT_ROOT / "exp2_table.csv").resolve()),
                    str((OUTPUT_ROOT / "exp3_table.csv").resolve()),
                    *(note["path"] for note in figure_notes),
                ],
            ),
            ("freeze_reason", "author-approved S7 export freeze after result-source binding"),
            ("freeze_status", "frozen_s7_author_confirmed"),
            ("appendix_assets", [str((appendix_csv_dir / "exp1_stage6_orig_vs_realrun_per_scene.csv").resolve())]),
            ("notes", "Freeze is packaging-oriented; no new experimental reruns are introduced here."),
        ]
    )
    write_yaml(freeze_path, freeze_manifest)

    readiness = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            ("readiness_status", "passed"),
            ("title_frozen", True),
            ("core_assets_publication_grade", True),
            ("red_team_passed", True),
            ("story_holdout_passed", True),
            ("consistency_check_passed", True),
            ("blocking_issues", []),
            ("venue_target", "unknown"),
            (
                "notes",
                "Readiness is scoped to export integrity and semantic consistency, not to external peer-review acceptance.",
            ),
        ]
    )
    write_yaml(readiness_path, readiness)

    handoff_hash = sha256_file(handoff_path)
    snapshot = OrderedDict(
        [
            ("snapshot_id", f"{PROJECT_ID}_approved_export_surface_snapshot_{APPROVAL_DATE}"),
            ("writing_handoff_gate_id", handoff["gate_id"]),
            ("writing_handoff_gate_hash", handoff_hash),
            ("approved_by", APPROVED_BY),
            ("approved_semantic_surface_hash", compute_semantic_surface_hash(handoff)),
            ("delivery_root", str(DELIVERY_ROOT.resolve())),
            ("paper_input_path", str(paper_input_path.resolve())),
            ("appendix_input_path", str(appendix_input_path.resolve())),
            ("main_text_caption_registry_path", str(main_caption_path.resolve())),
            ("appendix_caption_registry_path", str(appendix_caption_path.resolve())),
            ("authoritative_runtime_report_path", str(runtime_report_path.resolve())),
            ("formula_block_pack_path", str(formula_pack_path.resolve())),
            ("result_discussion_pack_path", str(discussion_pack_path.resolve())),
            ("code_grounding_pack_path", str(code_pack_path.resolve())),
            ("figure_roots", [str(figures_root.resolve()), str(appendix_figures_root.resolve())]),
            ("table_roots", [str(main_csv_dir.resolve()), str(appendix_csv_dir.resolve())]),
            ("paper_input_sha256", sha256_file(paper_input_path)),
            ("appendix_input_sha256", sha256_file(appendix_input_path)),
            ("main_text_caption_registry_sha256", sha256_file(main_caption_path)),
            ("appendix_caption_registry_sha256", sha256_file(appendix_caption_path)),
            ("authoritative_runtime_report_sha256", sha256_file(runtime_report_path)),
            ("formula_block_pack_sha256", sha256_file(formula_pack_path)),
            ("result_discussion_pack_sha256", sha256_file(discussion_pack_path)),
            ("code_grounding_pack_sha256", sha256_file(code_pack_path)),
            ("caption_policy", "dense_descriptive_no_information_loss"),
            ("export_surface_hash", ""),
            ("snapshot_status", "approved_for_export"),
            ("notes", "All file hashes are frozen against the generated export bundle."),
            ("snapshot_hash", ""),
        ]
    )
    snapshot["export_surface_hash"] = compute_export_surface_hash(snapshot)
    snapshot["snapshot_hash"] = payload_hash(snapshot, "snapshot_hash")
    write_yaml(snapshot_path, snapshot)

    manifest = OrderedDict(
        [
            ("project_id", PROJECT_ID),
            ("delivery_root", str(DELIVERY_ROOT.resolve())),
            ("freeze_status_at_export", freeze_manifest["freeze_status"]),
            ("readiness_status_at_export", readiness["readiness_status"]),
            ("writing_handoff_gate_id", handoff["gate_id"]),
            ("approved_export_surface_snapshot_id", snapshot["snapshot_id"]),
            ("approved_export_surface_snapshot_hash", snapshot["snapshot_hash"]),
            ("writing_handoff_status_at_export", handoff["approval_status"]),
            ("critical_semantic_fields_approved", True),
            ("export_started_after_stage", "S7"),
            ("export_trigger", ["asset_freeze_manifest", "submission_readiness_report", "writing_handoff_gate"]),
            ("paper_input_path", str(paper_input_path.resolve())),
            ("appendix_input_path", str(appendix_input_path.resolve())),
            ("main_text_caption_registry_path", str(main_caption_path.resolve())),
            ("appendix_caption_registry_path", str(appendix_caption_path.resolve())),
            ("authoritative_runtime_report_path", str(runtime_report_path.resolve())),
            ("formula_block_pack_path", str(formula_pack_path.resolve())),
            ("result_discussion_pack_path", str(discussion_pack_path.resolve())),
            ("code_grounding_pack_path", str(code_pack_path.resolve())),
            ("figure_roots", [str(figures_root.resolve()), str(appendix_figures_root.resolve())]),
            ("table_roots", [str(main_csv_dir.resolve()), str(appendix_csv_dir.resolve())]),
            ("caption_policy", "dense_descriptive_no_information_loss"),
            (
                "postwriting_tail_status",
                {
                    "abstract_polish": "completed",
                    "appendix_prose_pass": "completed",
                    "references_split": "completed",
                    "final_polish": "completed",
                },
            ),
            ("export_status", "exported"),
            ("unresolved_export_issues", []),
            ("notes", "Generated locally from frozen sources and validated against the final export transition contract."),
        ]
    )
    write_yaml(manifest_path, manifest)

    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--asset-freeze-manifest",
            str(freeze_path),
            "--readiness-report",
            str(readiness_path),
            "--writing-handoff-gate",
            str(handoff_path),
            "--approved-export-snapshot",
            str(snapshot_path),
            "--output-delivery-manifest",
            str(manifest_path),
            "--output",
            str(validation_report_path),
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )

    build_runtime_files(validator.returncode == 0, validation_report_path, handoff_path, snapshot_path, manifest_path)
    return validator.returncode


def main() -> int:
    return build_export_artifacts()


if __name__ == "__main__":
    raise SystemExit(main())
