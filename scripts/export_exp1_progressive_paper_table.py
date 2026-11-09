#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from statistics import mean


STAGE_ORDER = [
    "stage0_pgsr",
    "stage1_underwater",
    "stage2_bgclosure",
    "stage3_fargating",
    "stage4_capacity",
    "stage5_conflict",
    "stage6_pseudoclean",
]

STAGE_LABELS = {
    "stage0_pgsr": "Stage 0: PGSR",
    "stage1_underwater": "Stage 1: + Underwater",
    "stage2_bgclosure": "Stage 2: + BG Closure",
    "stage3_fargating": "Stage 3: + Far Gating",
    "stage4_capacity": "Stage 4: + Capacity Control",
    "stage5_conflict": "Stage 5: + Conflict-Aware BG",
    "stage6_pseudoclean": "Stage 6: + Pseudo-Clean",
}

STAGE_COMPONENTS = {
    "stage0_pgsr": "PGSR baseline",
    "stage1_underwater": "PGSR + underwater rendering",
    "stage2_bgclosure": "Stage1 + physically closed background",
    "stage3_fargating": "Stage2 + far-field depth gating",
    "stage4_capacity": "Stage3 + far-field densify blocking",
    "stage5_conflict": "Stage4 + BG conflict suppression/pruning",
    "stage6_pseudoclean": "Stage5 + pseudo-clean supervision",
}

SUMMARY_METRICS = [
    ("SSIM", "RGB SSIM", True),
    ("PSNR", "RGB PSNR", True),
    ("AIR_PSNR", "Clean PSNR", True),
    ("depth.rmse", "Depth RMSE", False),
    ("depth.absrel", "Depth AbsRel", False),
    ("normal.mean_angle_deg", "Normal Mean Angle", False),
]

FULL_METRICS = [
    "SSIM",
    "PSNR",
    "AIR_PSNR",
    "AIRBGBLACK_PSNR",
    "depth.rmse",
    "depth.absrel",
    "normal.mean_angle_deg",
]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def float_str(value, digits=3):
    return f"{value:.{digits}f}"


def parse_stage(model_name: str) -> str:
    return model_name.split("progressive_")[1]


def collect_records(root: Path):
    records = []
    for results_path in sorted(root.glob("*/exp1/*/results.json")):
        model_dir = results_path.parent
        scene_name = model_dir.parent.parent.name
        stage = parse_stage(model_dir.name)
        metrics = load_json(results_path)["ours_30000"]
        record = {
            "scene": scene_name,
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "components": STAGE_COMPONENTS[stage],
        }
        for key in FULL_METRICS:
            record[key] = float(metrics[key])
        records.append(record)
    return records


def summarize(records):
    summary = []
    for stage in STAGE_ORDER:
        stage_rows = [r for r in records if r["stage"] == stage]
        row = {
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "components": STAGE_COMPONENTS[stage],
            "num_scenes": len(stage_rows),
        }
        for key, _, _ in SUMMARY_METRICS:
            row[key] = mean(r[key] for r in stage_rows)
        summary.append(row)
    return summary


def filter_scene(records, scene_name: str):
    return [r for r in records if r["scene"] == scene_name]


def best_metric_map(summary_rows):
    best = {}
    for key, _, higher_is_better in SUMMARY_METRICS:
        values = [row[key] for row in summary_rows]
        target = max(values) if higher_is_better else min(values)
        best[key] = target
    return best


def write_csv(path: Path, rows, fieldnames):
    lines = [",".join(fieldnames)]
    for row in rows:
        values = []
        for name in fieldnames:
            value = row[name]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(summary_rows):
    best = best_metric_map(summary_rows)
    header = [
        "Stage",
        "Components",
        "Scenes",
        "RGB SSIM ↑",
        "RGB PSNR ↑",
        "Clean PSNR ↑",
        "Depth RMSE ↓",
        "Depth AbsRel ↓",
        "Normal Mean Angle ↓",
    ]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in summary_rows:
        cells = [
            row["stage_label"],
            row["components"],
            str(row["num_scenes"]),
        ]
        for key, _, _ in SUMMARY_METRICS:
            text = float_str(row[key], 3)
            if abs(row[key] - best[key]) <= 1e-9:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def analysis_paragraph(summary_rows):
    stage0 = next(row for row in summary_rows if row["stage"] == "stage0_pgsr")
    stage1 = next(row for row in summary_rows if row["stage"] == "stage1_underwater")
    stage2 = next(row for row in summary_rows if row["stage"] == "stage2_bgclosure")
    stage4 = next(row for row in summary_rows if row["stage"] == "stage4_capacity")
    stage5 = next(row for row in summary_rows if row["stage"] == "stage5_conflict")
    stage6 = next(row for row in summary_rows if row["stage"] == "stage6_pseudoclean")

    return (
        "表 X 给出了当前 6 个 FUNA `blue-4` 场景上的递进式消融结果。整体趋势表明，"
        "该实验更支持“逐步减少 geometry-medium misattribution”的链式叙事，而不是某一单独模块对所有指标的统一提升。"
        f"以 PGSR 基线为起点，`stage1_underwater` 首先将平均 clean PSNR 从 {stage0['AIR_PSNR']:.3f} 提升到 {stage1['AIR_PSNR']:.3f}，"
        f"同时将平均法线误差从 {stage0['normal.mean_angle_deg']:.3f} 降至 {stage1['normal.mean_angle_deg']:.3f}，"
        "说明水下成像建模主要改善了外观解释。进一步引入背景闭合后，平均 depth RMSE 从 "
        f"{stage1['depth.rmse']:.3f} 显著降至 {stage2['depth.rmse']:.3f}，"
        "表明 no-hit 与远场区域获得了更稳定的物理解释。随后 far gating 与 capacity control 继续带来稳定收益，"
        f"其中 `stage4_capacity` 在平均 RGB PSNR 上达到最高值 {stage4['PSNR']:.3f}。"
        f"最显著的几何改进出现在 `stage5_conflict`，其平均 depth RMSE 进一步降至 {stage5['depth.rmse']:.3f}，"
        f"depth AbsRel 降至 {stage5['depth.absrel']:.3f}，说明背景冲突抑制与证据剪枝有效减少了介质诱导的伪几何。"
        f"最后加入 pseudo-clean supervision 后，`stage6_pseudoclean` 仍相对基线保持明显优势"
        f"（例如 clean PSNR {stage6['AIR_PSNR']:.3f} vs. {stage0['AIR_PSNR']:.3f}，"
        f"depth RMSE {stage6['depth.rmse']:.3f} vs. {stage0['depth.rmse']:.3f}），"
        "但并未在所有指标上继续单调提升。"
        "这说明 pseudo-clean supervision 更偏向外观正则，而几何主导收益主要来自背景闭合、容量控制与 conflict-aware 背景约束。"
    )


def scene_analysis(scene_name: str, summary_rows):
    stage0 = next(row for row in summary_rows if row["stage"] == "stage0_pgsr")
    stage1 = next(row for row in summary_rows if row["stage"] == "stage1_underwater")
    stage2 = next(row for row in summary_rows if row["stage"] == "stage2_bgclosure")
    stage3 = next(row for row in summary_rows if row["stage"] == "stage3_fargating")
    stage4 = next(row for row in summary_rows if row["stage"] == "stage4_capacity")
    stage5 = next(row for row in summary_rows if row["stage"] == "stage5_conflict")
    stage6 = next(row for row in summary_rows if row["stage"] == "stage6_pseudoclean")

    caption = (
        f"Table X. Progressive ablation on `{scene_name}`. Starting from PGSR, we progressively add underwater rendering, "
        "physically closed background modeling, far-field gating, capacity control, conflict-aware background suppression, "
        "and pseudo-clean supervision. The scene clearly shows that different modules improve different parts of the "
        "geometry-medium disentanglement pipeline, rather than producing a uniform gain from a single design choice."
    )

    body = (
        f"在 `{scene_name}` 上，递进式消融呈现出非常清晰的模块分工。首先，`stage1_underwater` 将 clean PSNR 从 "
        f"{stage0['AIR_PSNR']:.3f} 提升到 {stage1['AIR_PSNR']:.3f}，同时 RGB PSNR 也从 {stage0['PSNR']:.3f} 提升到 "
        f"{stage1['PSNR']:.3f}，但 depth RMSE 却从 {stage0['depth.rmse']:.3f} 上升到 {stage1['depth.rmse']:.3f}，"
        "说明仅引入水下成像建模主要改善的是外观解释，而非几何恢复。随后，背景闭合立即将 depth RMSE 大幅降至 "
        f"{stage2['depth.rmse']:.3f}，表明 no-hit 与远场区域获得了更稳定的物理解释。继续加入 far gating 与 "
        f"capacity control 后，depth RMSE 进一步从 {stage2['depth.rmse']:.3f} 降到 {stage3['depth.rmse']:.3f} 再到 "
        f"{stage4['depth.rmse']:.3f}，同时 RGB PSNR 在 `stage4_capacity` 达到全场最高值 {stage4['PSNR']:.3f}。"
        f"最显著的几何修正出现在 `stage5_conflict`，其 depth RMSE 进一步降至 {stage5['depth.rmse']:.3f}，说明背景冲突抑制与证据剪枝有效清除了介质诱导的伪结构。"
        f"最后，`stage6_pseudoclean` 相比基线仍保持明显优势，例如 clean PSNR 从 {stage0['AIR_PSNR']:.3f} 提升到 "
        f"{stage6['AIR_PSNR']:.3f}，depth RMSE 从 {stage0['depth.rmse']:.3f} 降到 {stage6['depth.rmse']:.3f}；"
        f"但它并未继续优于 `stage5_conflict` 的最优几何结果（{stage6['depth.rmse']:.3f} vs. {stage5['depth.rmse']:.3f}）。"
        "这一现象说明 pseudo-clean supervision 更偏向外观正则，而真正主导几何改进的是背景闭合、远场容量控制与 conflict-aware 背景约束。"
    )
    return caption, body


def main():
    parser = argparse.ArgumentParser(description="Export paper-ready exp1 progressive ablation tables.")
    parser.add_argument(
        "--root",
        default="/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu",
        help="Root directory containing <scene>/exp1/*/results.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/new/home_old/ycy/UW_PGSR_4.8.2/outputs/funa_exp13_6gpu",
        help="Directory to save markdown/csv/text outputs",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(root)
    if len(records) != 42:
        print(f"[warn] expected 42 exp1 results, found {len(records)}")

    summary_rows = summarize(records)

    summary_csv_rows = []
    for row in summary_rows:
        summary_csv_rows.append(
            {
                "stage": row["stage"],
                "stage_label": row["stage_label"],
                "components": row["components"],
                "num_scenes": row["num_scenes"],
                "rgb_ssim": row["SSIM"],
                "rgb_psnr": row["PSNR"],
                "clean_psnr": row["AIR_PSNR"],
                "depth_rmse": row["depth.rmse"],
                "depth_absrel": row["depth.absrel"],
                "normal_mean_angle_deg": row["normal.mean_angle_deg"],
            }
        )

    summary_csv_path = output_dir / "exp1_progressive_blue4_paper_table.csv"
    summary_md_path = output_dir / "exp1_progressive_blue4_paper_table.md"
    full_csv_path = output_dir / "exp1_progressive_blue4_full_results.csv"
    analysis_path = output_dir / "exp1_progressive_blue4_analysis.md"
    bay_scene_name = "Bay_submerged_ships_2"
    bay_summary_csv_path = output_dir / "bay_submerged_ships_2_exp1_ablation_table.csv"
    bay_summary_md_path = output_dir / "bay_submerged_ships_2_exp1_ablation_table.md"
    bay_analysis_path = output_dir / "bay_submerged_ships_2_exp1_ablation_analysis.md"

    write_csv(
        summary_csv_path,
        summary_csv_rows,
        [
            "stage",
            "stage_label",
            "components",
            "num_scenes",
            "rgb_ssim",
            "rgb_psnr",
            "clean_psnr",
            "depth_rmse",
            "depth_absrel",
            "normal_mean_angle_deg",
        ],
    )

    write_csv(
        full_csv_path,
        records,
        [
            "scene",
            "stage",
            "stage_label",
            "components",
            "SSIM",
            "PSNR",
            "AIR_PSNR",
            "AIRBGBLACK_PSNR",
            "depth.rmse",
            "depth.absrel",
            "normal.mean_angle_deg",
        ],
    )

    md = [
        "# Exp1 Progressive Ablation Table",
        "",
        "All values are averages over the 6 currently available FUNA `blue-4` scene runs (`6 scenes x 7 stages = 42 results.json`).",
        "",
        markdown_table(summary_rows),
    ]
    summary_md_path.write_text("\n".join(md), encoding="utf-8")
    analysis_path.write_text(analysis_paragraph(summary_rows) + "\n", encoding="utf-8")

    bay_records = filter_scene(records, bay_scene_name)
    bay_summary_rows = summarize(bay_records)
    bay_summary_csv_rows = []
    for row in bay_summary_rows:
        bay_summary_csv_rows.append(
            {
                "stage": row["stage"],
                "stage_label": row["stage_label"],
                "components": row["components"],
                "rgb_ssim": row["SSIM"],
                "rgb_psnr": row["PSNR"],
                "clean_psnr": row["AIR_PSNR"],
                "depth_rmse": row["depth.rmse"],
                "depth_absrel": row["depth.absrel"],
                "normal_mean_angle_deg": row["normal.mean_angle_deg"],
            }
        )

    write_csv(
        bay_summary_csv_path,
        bay_summary_csv_rows,
        [
            "stage",
            "stage_label",
            "components",
            "rgb_ssim",
            "rgb_psnr",
            "clean_psnr",
            "depth_rmse",
            "depth_absrel",
            "normal_mean_angle_deg",
        ],
    )

    bay_md = [
        f"# {bay_scene_name} Exp1 Single-Scene Ablation",
        "",
        f"All values come from the 7 progressive stages of `{bay_scene_name}_blue_4`.",
        "",
        markdown_table(bay_summary_rows).replace("| Scenes |", "|").replace(" | Scenes", "").replace(" | 1 ", " "),
    ]
    # Rebuild a clean single-scene table without the redundant Scenes column.
    bay_header = [
        "Stage",
        "Components",
        "RGB SSIM ↑",
        "RGB PSNR ↑",
        "Clean PSNR ↑",
        "Depth RMSE ↓",
        "Depth AbsRel ↓",
        "Normal Mean Angle ↓",
    ]
    best = best_metric_map(bay_summary_rows)
    bay_lines = []
    bay_lines.append("| " + " | ".join(bay_header) + " |")
    bay_lines.append("|" + "|".join(["---"] * len(bay_header)) + "|")
    for row in bay_summary_rows:
        cells = [row["stage_label"], row["components"]]
        for key, _, _ in SUMMARY_METRICS:
            text = float_str(row[key], 3)
            if abs(row[key] - best[key]) <= 1e-9:
                text = f"**{text}**"
            cells.append(text)
        bay_lines.append("| " + " | ".join(cells) + " |")
    bay_summary_md_path.write_text(
        "\n".join(
            [
                f"# {bay_scene_name} Exp1 Single-Scene Ablation",
                "",
                f"All values come from the 7 progressive stages of `{bay_scene_name}_blue_4`.",
                "",
                "\n".join(bay_lines),
                "",
            ]
        ),
        encoding="utf-8",
    )

    bay_caption, bay_body = scene_analysis(bay_scene_name, bay_summary_rows)
    bay_analysis_path.write_text(
        "\n".join(
            [
                "## Caption",
                "",
                bay_caption,
                "",
                "## Body",
                "",
                bay_body,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] summary csv: {summary_csv_path}")
    print(f"[done] summary md: {summary_md_path}")
    print(f"[done] full csv: {full_csv_path}")
    print(f"[done] analysis md: {analysis_path}")
    print(f"[done] bay csv: {bay_summary_csv_path}")
    print(f"[done] bay md: {bay_summary_md_path}")
    print(f"[done] bay analysis md: {bay_analysis_path}")


if __name__ == "__main__":
    main()
