# a_configs 配置说明

本目录是 ISCKD 的**自定义实验配置区**，与官方 `configs/` 平行存放，共 **67 个 `.py`**。
仓库母集为 mmdetection3d `1.0.0.dev0` 快照（`mmdet3d/version.py` 仍写作 `0.17.2`，属 0.17.2 之后、mmengine 之前的 master 分支），因此同时兼容旧式 `pretrained=` 与新式 `init_cfg=` 两种权重写法。

---

## 1. 目录结构

```
a_configs/
├── dairv2x/mvxnet/          3  DAIR-V2X 多模态检测（数据已转 KITTI 格式）
├── kitti/
│   ├── 3dssd/               4  点云单阶段基线 SSD3DNet
│   ├── PGD_OPA/             5  paper3：PGD + OPA 进化分区指派（PGD_GA_HEAD）
│   ├── PGD_big/             6  paper5：大 backbone 教师/上限模型（RegNet、Swin）
│   ├── PGD_mini/           25  paper5：轻量学生 + 蒸馏（DIS_PGD / PGDHeadDIS）
│   ├── mvxnet/              2  KITTI 多模态基线 MVX-Net
│   └── smoke/              11  SMOKE 轻量单目 3D + 蒸馏（DIS_SMOKE）
└── kitti_old/              11  mmdetection3d-0.17 的 a_configs/kitti 原样归档
    ├── 3dssd/               4
    ├── PGD/                 4
    └── fcos3d/              3
```

`kitti_old/` 于 2026-11-20 从 0.17 仓逐字节迁入，未做任何改写，仅作历史对照；新实验请使用 `kitti/` 下各目录。

---

## 2. 实验线与论文对应

| 目录                                 | 论文/用途        | 检测器                          | 检测头                         | 说明                                                                                                       |
| ------------------------------------ | ---------------- | ------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `kitti/PGD_OPA`                    | **paper3** | `FCOSMono3D`                  | `PGDHead` → `PGD_GA_HEAD` | 把 OPA 的进化算法分区指派移植到单目 3D（geatpy 拟合 cls/reg 损失平面上的最优分界线，替代 PAA 的 GMM 阈值） |
| `kitti/PGD_big`                    | **paper5** | `FCOSMono3D`                  | `PGDHead`                    | 大 backbone（RegNetX-3.2GF、Swin-T/S/B/L），用作蒸馏教师或精度上限                                         |
| `kitti/PGD_mini`                   | **paper5** | `FCOSMono3D` / `DIS_PGD`    | `PGDHead` / `PGDHeadDIS`   | 轻量学生（MobileNetV2、ResNet18、RegNetX-800MF、ShuffleNet、Swin-T/S）+ 冻结教师蒸馏                       |
| `kitti/smoke`                      | **paper5** | `SMOKEMono3D` / `DIS_SMOKE` | `SMOKEMono3DHead`            | SMOKE/DLANet 线的轻量化与蒸馏                                                                              |
| `kitti/mvxnet`、`dairv2x/mvxnet` | 多模态基线       | `DynamicMVXFasterRCNN`        | `Anchor3DHead`               | 图像+点云融合；DAIR-V2X 以 KITTI 格式读取                                                                  |
| `kitti/3dssd`                      | 点云基线         | `SSD3DNet`                    | `SSD3DHead`                  | PointNet2SAMSG 骨干，单阶段点云检测                                                                        |
| `kitti_old/*`                      | 0.17 归档        | 同上                            | 同上                           | 历史配置，见 §4.7                                                                                         |

---

## 3. 命名约定

| 记号                                  | 含义                                                                                                                                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dis_` 前缀                         | 蒸馏配置：检测器换成 `DIS_PGD`/`DIS_SMOKE`，内含冻结教师 `tea_model` + 教师权重 `tea_pretrained`，学生头导出 `loss_stu` 供教师侧对齐                                                                            |
| `Dis_PGD_<学生>_<教师>`             | 教师后缀：**无后缀 = R101**（`pgd_101_2`）、**`_reg` = RegNetX-3.2GF**（`PGD_reg3.2`）、**`_swint`/`_swinl` = Swin-T/Swin-L**                                                                 |
| `_h`                                | 半宽。PGD 线：FPN `out_channels` 与 head `feat_channels` 256→128；smoke 线：head `cls_branch`/`reg_branch` 256→128（`smoke_h` 还把 `in_channels`/`feat_channels` 64→32，见根 README「已知问题」第 4 条） |
| `_q`                                | 四分之一宽。`smoke_q`：head branch 256→64、`feat_channels` 64→16                                                                                                                                                    |
| `smoke<数字>`                       | DLANet 深度：`smoke34`/`smoke60`/`smoke102`                                                                                                                                                                         |
| `Dis_smoke_<教师depth>_<学生depth>` | 如 `Dis_smoke_102_34` = 教师 DLA102（名义）、学生 DLA34                                                                                                                                                                 |
| backbone 缩写                         | `mobv2`=MobileNetV2，`r18`=ResNet18，`reg800`=RegNetX-800MF，`reg3.2`=RegNetX-3.2GF，`shuf`=ShuffleNet，`swint/swins/swinb/swinl`=Swin-T/S/B/L                                                                |

---

## 4. 各目录接线与训练设置

### 4.1 kitti/PGD_OPA（5 个，paper3）

继承链：`fcos3d.py` + `kitti-mono3d.py` → `pgd.py` → `pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py` → `pgd_ga.py`

| 文件                                                  | 作用                                                                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `fcos3d.py`                                         | FCOSMono3D 模型基类（nuScenes 风格：`num_classes=10`、`pred_attrs=True`、`pred_velo=True`，KITTI 入口会覆盖为 3 类） |
| `kitti-mono3d.py`                                   | 数据/流水线基类：`data_root='/home/data/kitti/'`，`samples_per_gpu=2`，SGD lr=0.02、12ep（入口覆盖）                   |
| `pgd.py`                                            | 换 `PGDHead` + `PGDBBoxCoder`（`code_size=9`、`diff_rad_by_sin=True`）                                             |
| `pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d.py` | R101-caffe 入口：bs=6、48ep、lr=0.001、step[32,44]、`work_dir="work_dirs/pgd_101_4"`                                     |
| `pgd_ga.py`                                         | **paper3 入口**：仅把 head 换成 `PGD_GA_HEAD`，`checkpoint_config interval=4`，`work_dir='work_dirs/demo'`     |

`PGD_GA_HEAD`（`mmdet3d/models/dense_heads/pgd_ga_head.py`）依赖 **geatpy**，在 `ga_reassign` 中用进化算法拟合分区线；运行前需额外安装 geatpy。

### 4.2 kitti/PGD_big（6 个，paper5 教师/上限）

5 个入口 + `kitti-mono3d.py`（`data_root='/home/airstudio/data/KITTI_new/'`）。全部 `weight_dim=1`、48ep、lr=0.001、step[32,44]。

| 配置              | backbone        | 关键参数                     | bs |
| ----------------- | --------------- | ---------------------------- | -- |
| `PGD_reg3.2.py` | RegNet          | `regnetx_3.2gf`            | 9  |
| `PGD_swint.py`  | SwinTransformer | embed 96, depths [2,2,6,2]   | 9  |
| `PGD_swins.py`  | SwinTransformer | embed 96, depths [2,2,18,2]  | 6  |
| `PGD_swinb.py`  | SwinTransformer | embed 128, depths [2,2,18,2] | 6  |
| `PGD_swinl.py`  | SwinTransformer | embed 192, depths [2,2,18,2] | 3  |

> ⚠ 这 5 个配置都命中 `pgd_head.py` 的空 GT 分支 bug（根 README「已知问题」第 1 条）。Swin 骨干来自 `mmdet3d/models/backbones/swin.py`（官方较新版本拷贝，来源待确认）。

### 4.3 kitti/PGD_mini（25 个，paper5 学生+蒸馏）

`kitti-mono3d.py`（`data_root='/home/airstudio/data/KITTI_new/'`）+ 8 个普通学生 + 16 个蒸馏配置。全部 48ep、lr=0.001、step[32,44]；学生与教师头均为 `PGDHeadDIS`。

**普通学生（8）**：4 种 backbone × {标准, `_h` 半宽}

| backbone                                           | 标准              | 半宽                | bs |
| -------------------------------------------------- | ----------------- | ------------------- | -- |
| MobileNetV2（`out_indices=(1,2,4,7)`）           | `PGD_mobv2.py`  | `PGD_mobv2_h.py`  | 9  |
| ResNet18                                           | `PGD_r18.py`    | `PGD_r18_h.py`    | 3  |
| RegNetX-800MF                                      | `PGD_reg800.py` | `PGD_reg800_h.py` | 9  |
| ShuffleNet（自定义注册，FPN in=[48,480,960,1920]） | `PGD_shuf.py`   | `PGD_shuf_h.py`   | 9  |

**蒸馏配置（16）**：`DIS_PGD` = 冻结教师 + 学生

| 学生          | 教师 R101（`pgd_101_2`）         | 教师 RegNet（`PGD_reg3.2`）           | 教师 Swin-T                                   | 教师 Swin-L             |
| ------------- | ---------------------------------- | --------------------------------------- | --------------------------------------------- | ----------------------- |
| MobileNetV2   | `Dis_PGD_mobv2`                  | `Dis_PGD_mobv2_reg`                   | `Dis_PGD_mobv2_swint`                       | `Dis_PGD_mobv2_swinl` |
| ResNet18      | `Dis_PGD_r18`、`Dis_PGD_r18_h` | —                                      | `Dis_PGD_r18_swint`、`Dis_PGD_r18h_swint` | —                      |
| RegNetX-800MF | —                                 | `Dis_PGD_reg800`、`Dis_PGD_reg800h` | —                                            | —                      |
| ShuffleNet    | `Dis_PGD_shuf`                   | `Dis_PGD_shuf_reg`                    | `Dis_PGD_shuf_swint`                        | `Dis_PGD_shuf_swinl`  |
| Swin-T        | —                                 | —                                      | —                                            | `Dis_PGD_swint_swinl` |
| Swin-S        | —                                 | —                                      | —                                            | `Dis_PGD_swins_swinl` |

教师权重路径（两套云端根目录）：

- R101 / RegNet：`/workspace/mmdetection3d-master/work_dirs/{pgd_101_2, PGD_reg3.2}/epoch_48.pth`
- Swin-T / Swin-L：`/home/airstudio/code/mmdet3d-1.0/work_dirs/{PGD_swint, PGD_swinl}/epoch_48.pth`

> `_reg`/`_swint`/`_swinl` 后缀标识**教师身份**，不是损失函数变体；教师 backbone/neck 在配置内完整重建（R101+FPN 或 RegNet+FPN 或 Swin+FPN）。

### 4.4 kitti/smoke（11 个）

`kitti-mono3d.py`（`data_root='/home/data/kitti/'`）+ 5 个普通 + 5 个蒸馏。全部 **Adam lr=2.5e-4、72ep、step[50]、bs=8**，`find_unused_parameters=True`。

**普通（5）**

| 配置            | DLA depth | head in/feat | cls/reg branch |
| --------------- | --------- | ------------ | -------------- |
| `smoke34.py`  | 34        | 64 / 64      | 256            |
| `smoke60.py`  | 60        | 128 / 128    | 256            |
| `smoke102.py` | 102       | 128 / 128    | 256            |
| `smoke_h.py`  | 34        | 32 / 32      | 128            |
| `smoke_q.py`  | 34        | 64 / 16      | 64             |

**蒸馏（5）**：`DIS_SMOKE`，教师权重统一在 `/workspace/mmdetection3d-master/work_dirs/`

| 配置                          | 学生                               | 教师结构          | 加载权重                  | 结构/权重是否一致                                                 |
| ----------------------------- | ---------------------------------- | ----------------- | ------------------------- | ----------------------------------------------------------------- |
| `Dis_smoke_60_34.py`        | DLA34                              | DLA60             | `smoke60/epoch_72.pth`  | ✅                                                                |
| `Dis_smoke_102_34.py`       | DLA34                              | **DLA60**   | `smoke102/epoch_72.pth` | ❌ 错配（根 README「已知问题」第 2 条）                           |
| `Dis_smoke_102_60.py`       | DLA60                              | **DLA60**   | `smoke102/epoch_72.pth` | ❌ 错配（根 README「已知问题」第 2 条）                           |
| `Dis_smoke_h.py`            | DLA34，branch 128（in/feat 仍 64） | DLA34，branch 256 | `smoke/epoch_72.pth`    | ✅（学生口径与 `smoke_h` 不同，见根 README「已知问题」第 4 条） |
| `Dis_smoke_res101_dla34.py` | DLA34                              | DLA60             | `smoke60/epoch_72.pth`  | ⚠ 文件名中的 res101 无对应                                       |

### 4.5 kitti/mvxnet（2）与 dairv2x/mvxnet（3）

- 模型：`DynamicMVXFasterRCNN` + `Anchor3DHead` + ResNet（18/50）+ `DynamicVFE`/`PointFusion`。
- 调度：**AdamW lr=0.003、CosineAnnealing（warmup linear 1000 iter、ratio 0.1、min_lr_ratio 1e-5）、40ep、bs=2**，grad_clip max_norm=35。
- `kitti/mvxnet/{mvxnet18,mvxnet50}.py`：`data_root='/home/data/kitti/'`。
- `dairv2x/mvxnet/mvx50.py`：**自包含**（不继承 kitti 基类），`dataset_type="KittiDataset"`、`data_root="/home/data/dair-v2x-v/"`、`ann_file=kitti_infos_{train,val}.pkl` —— 即 DAIR-V2X 已离线转成 KITTI 格式；`img_scale=(960,540)`、`voxel_size=[0.05,0.05,0.1]`、`point_cloud_range=[0,-40,-3,70.4,40,1]`，类别 `["Pedestrian","Cyclist","Car"]`。
- `dairv2x/mvxnet/{mvxnet18,mvxnet50}.py`：同数据集的 ResNet18/50 变体。

### 4.6 kitti/3dssd（4）

`3dssd.py`（`SSD3DNet` + `SSD3DHead` + `PointNet2SAMSG`）、`kitti-3d-car.py`（`data_root='/home/data/KITTI_new/'`）、`default_runtime.py`、入口 `3dssd_4x4_kitti-3d-car.py`（**AdamW lr=0.002、80ep、bs=4**）。

### 4.7 kitti_old（11，0.17 原样归档）

| 子目录           | 内容                                                                                                                                                                                                                             | 在本仓（dev0 母集）的可运行性                                                                                                                                                                                                                                     |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PGD/`（4）    | `fcos3d.py`、`kitti-mono3d.py`、`pgd.py`、`pgd_r101_...py`（bs=3）                                                                                                                                                       | 在 0.17 中是**死配置**（`PGDHead` 只存在于 dev0）；迁入本仓后可构建。`pgd.py` **没有** `diff_rad_by_sin`（与 `PGD_OPA/pgd.py` 的差异，归档原样保留）；`fcos3d.py` 用旧式 `pretrained=` 且指向 `open-mmlab://detectron2/resnet101_caffe` |
| `fcos3d/`（3） | 入口 `fcos_3d_kitti_mono3d.py`：`num_classes=3`、`pred_attrs=False`、`pred_velo=False`、`bbox_code_size=7`、bs=3、lr=0.001（bias_lr_mult=2）、grad_clip 35、48ep、step[32,44]、`work_dirs='./work_dirs/fcos3d_demo'` | 旧式 `pretrained='/home/pretrained model/resnet101_caffe-3ad79236.pth'`，本仓兼容该 API                                                                                                                                                                         |
| `3dssd/`（4）  | 与 `kitti/3dssd` 同构，但 `data_root='/home/airstudio/data/KITTI_new/'`、bs=8                                                                                                                                                | 可构建，路径需本地化                                                                                                                                                                                                                                              |

---

## 5. 数据与权重路径

本地运行前必须改写以下三类路径；配置中**没有**任何相对路径兜底。

**数据根目录**

| data_root                           | 涉及目录                                                 |
| ----------------------------------- | -------------------------------------------------------- |
| `/home/data/kitti/`               | `PGD_OPA`、`smoke`、`kitti/mvxnet`                 |
| `/home/airstudio/data/KITTI_new/` | `PGD_big`、`PGD_mini`                                |
| `/home/data/KITTI_new/`           | `kitti/3dssd`、`kitti_old/PGD`、`kitti_old/fcos3d` |
| `/home/airstudio/data/KITTI_new/` | `kitti_old/3dssd`                                      |
| `/home/data/dair-v2x-v/`          | `dairv2x/mvxnet`                                       |

**预训练权重**（两个根目录：`/home/pretrained model/`、`/home/airstudio/pretrained model/`）

`resnet18-5c106cde.pth`、`resnet50_caffe-788b5fa3.pth`、`resnet101_caffe-3ad79236.pth`、`RegNetX-3.2GF.pth`、`RegNetX-800MF.pth`、`dla34-ba72cf86.pth`、`dla60-24839fc4.pth`、`dla102-d94d9790.pth`、`swin_{tiny,small,base,large}_patch4_window7_224[_22k].pth`；另有 1 处 `open-mmlab://detectron2/resnet101_caffe`（`kitti_old/PGD/fcos3d.py`，需联网或改本地）。

**教师权重 `tea_pretrained`**：见 §4.3、§4.4 表格（`/workspace/mmdetection3d-master/work_dirs/...` 与 `/home/airstudio/code/mmdet3d-1.0/work_dirs/...`）。
