# UW_PGSR / GeoMed-GS 代码结构

本说明依据当前 `train.py`、`arguments/__init__.py`、渲染器及本地 Method 写作材料核对。实际执行行为以源码和运行时参数为准。

## 主流程

```text
水下 RGB + COLMAP 相机与稀疏点云
    → Scene / Camera / GaussianModel
    → 平面高斯光栅化：介质前外观 J、alpha、深度、法线
    → CompleteUnderwaterPhysics：衰减 A、后向散射 B、远场颜色 B_inf
    → 命中判断、远深度门控与强 alpha 保护、背景闭合
    → 水下 RGB 重建与正则化
    → 高斯参数 / 外观模型 / 水体参数联合优化
    → 远区域增密过滤、背景不透明度抑制、多视角冲突剪枝
    → 保存模型 → render.py → metrics.py / metrics3.py
```

`air_image` 表示施加水体成像前的模型外观。默认训练输入是水下 RGB 与相机；同名伪清晰图像存在时，可以额外提供弱监督。它不等同于默认使用配对真实清晰图训练。

## 核心源码

| 文件或目录 | 作用 |
| --- | --- |
| `train.py` | 训练入口；背景 mask、损失、优化、增密过滤、背景剪枝、验证与保存 |
| `arguments/__init__.py` | 数据、渲染、优化参数和默认值；同时供训练与渲染读取 |
| `scene/__init__.py` | 场景初始化、训练/测试相机、邻近视角、模型保存 |
| `scene/dataset_readers.py`、`scene/colmap_loader.py` | COLMAP/Blender 数据读取；COLMAP 支持 `sparse/0` 与 `sparse` |
| `scene/cameras.py`、`utils/camera_utils.py` | 相机、图像读取、伪清晰图路径、可选深度/法线输入 |
| `scene/gaussian_model.py` | 高斯参数、优化器、clone/split、剪枝、不透明度重置及统计维护 |
| `scene/app_model.py` | 视角外观模型 |
| `scene/complete_physics_model.py` | `BackscatterNet`、`AttenuationNet`、`CompleteUnderwaterPhysics`，物理正则与权重存取 |
| `gaussian_renderer/__init__.py` | 高斯光栅化、水体成像、命中深度路由、背景闭合与中间量返回 |
| `utils/loss_utils.py`、`utils/graphics_utils.py` | 图像/几何损失、图像梯度、邻域 patch 投影与 warp |
| `lpipsPyTorch/` | 感知损失和评估实现 |
| `submodules/diff-plane-rasterization/` | 平面高斯 CUDA/C++ 光栅化扩展与 GLM 源码 |
| `submodules/simple-knn/` | 高斯初始化所需 CUDA/C++ 邻域距离扩展 |

这里的 `submodules/` 内容已经直接纳入 Git 跟踪，不需要用本机编译的 `.so` 或 `build/` 目录替代源码。

## 当前有效机制与兼容代码

当前主线启用完整水体模型、hit-aware depth、远场背景闭合、自适应远深度阈值、远距离强 alpha 保护、背景一致性、背景不透明度抑制、远区域增密过滤，以及保守背景剪枝。PGSR 的几何正则、多视角可见性清理和 opacity reset 继续使用。

`MediumEstimator`、DRO、Metric3D、旧版 physics-aware densify/pruning 等实现仍有兼容代码。它们不能仅凭函数存在就被视为当前有效方法：训练主路径优先使用 `CompleteUnderwaterPhysics`；DRO 和 Metric3D 的默认设置关闭，`lambda_far_alpha` 默认是 0。

伪清晰监督需要特别看真实分支：当前 `train.py` 根据 `lambda_fake_air`、`lambda_fake_air_lap`、`lambda_fake_air_lpips` 和图像是否存在决定是否计算，不能只看 `use_fake_air_supervision` 的布尔默认值。要关闭该监督，需要将这三个权重都设为 0。

## 渲染、评估与诊断

| 文件 | 作用 |
| --- | --- |
| `render.py` | 加载场景与水体权重；导出水下 RGB、介质前外观、深度/法线 EXR，可进行网格提取 |
| `metrics.py` | RGB、深度、法线和清晰外观评估 |
| `metrics3.py` | 扩展评估：前景指标、清晰外观、UIQM/UCIQE/NIQE；NIQE 依赖可选 `pyiqa` |
| `tools/check_masks.py` | 检查 GT + alpha + B_inf 背景 mask 与剪枝候选 |
| `tools/export_pipeline_intermediates.py` | 导出渲染中间量，帮助核对背景闭合和物理成像 |
| `depth_vis_exr.py`、`exr_normals_to_png.py` | 深度/法线 EXR 可视化 |
| `prep_depth_gt_align.py` | 深度 GT 对齐辅助工具 |
| `render_air_bgblack_batch.sh`、`batch_eval_metrics3.sh` | 批量黑背景外观导出与扩展指标评估；运行前核对路径、GPU 与模型目录 |

## 实验与辅助脚本

| 文件或文件组 | 用途及当前状态 |
| --- | --- |
| `run_exp1.sh`、`run_exp2.sh`、`run_exp3.sh` | 递进消融、场景分组、伪清晰监督对比的历史编排脚本 |
| `run_funa_exp13_6gpu*.sh` | FUNA 多 GPU、补跑、第三阶段及渲染/评估编排，包含本机路径与卡号 |
| `scripts/analyze_experiments.py`、`scripts/scene_group_analysis.py` | 汇总已有实验、按场景分组统计 |
| `scripts/export_exp123_tables.py`、`scripts/export_exp1_progressive_paper_table.py` | 从已有结果导出实验表格 |
| `scripts/build_funa_final_delivery.py` | 已有 FUNA 结果的材料导出工具，包含项目本地约定 |
| `scripts/gen_depth_and_params.py`、`scripts/precompute_metric3d_normals.py` | 可选旧实验的深度/Metric3D 法线预处理，需要外部仓库和模型权重 |
| `scripts/preprocess/` | COLMAP、自定义数据、DTU、TnT 数据预处理 |
| `scripts/run_dtu.py`、`scripts/run_tnt.py`、`scripts/run_mip360.py`、`full_eval.py` | 上游 PGSR 数据集编排入口；使用前按当前数据布局核对 |
| `scripts/eval_dtu/`、`scripts/tnt_eval/` | 上游 DTU/TnT 几何评估 |
| `scene.sh`、`test1.sh`、`test2.sh`、`test3.sh` | 本地命令记录/运行示例，包含历史路径或参数 |
| `patch_train.py` | 历史训练脚本修补工具，会修改 `train.py`，不属于正常训练启动步骤 |

部分历史编排脚本传入 `--experiment_profile`，但当前 `train.py` 和 `arguments/__init__.py` 没有定义该参数，因此不能直接用这些脚本启动当前版本。这里保留原始脚本供追溯；当前可用入口请先采用 README 的单场景命令。此问题不能靠删除参数解决，因为各 profile 对应的消融配置也需要明确还原。

## 代码仓库范围

源码、CUDA 扩展源码、实验/评估/可视化脚本和代码说明纳入版本管理。训练数据、`outputs/`、`output/`、权重、日志、Python 缓存、本机编译产物、备份以及本地论文草稿/写作材料通过 `.gitignore` 排除，保留在本地。

`research_os/` 是独立的本地科研工作流包，不参与上述训练依赖链，本次代码推送未纳入。许可证与上游代码署名保持保留，使用范围见根目录及各第三方目录中的许可证。
