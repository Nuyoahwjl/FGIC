# FGIC 训练与测试完整指南（WebFG-400 / WebiNat-5000）

本项目是一个面向 **FGIC（Fine-Grained Image Classification，细粒度图像分类）** 的 Docker 化复现工程，核心目标是在 **WebFG-400** 与 **WebiNat-5000** 两个数据集上完成：

- ConvNeXtv2 教师模型训练
- DINOv2 主模型训练（知识蒸馏）
- 测试集推理与 CSV 结果导出

---

## 1. 项目技术路线

### 1.1 总体思路

项目采用两阶段训练策略：

1. **教师模型阶段（Teacher）**
   - 模型：`convnextv2_large`
   - 数据：WebFG-400 / WebiNat-5000 各训练一个教师模型
2. **最终模型阶段（Main / Student）**
   - 模型：`dinov2_large`
   - 训练时加载对应教师模型权重，使用蒸馏损失 + 监督损失联合优化

### 1.2 关键训练机制

- 多 GPU 训练：PyTorch DDP（自动按可见 GPU 数启动）
- 鲁棒监督：`GCE_loss`
- 知识蒸馏：
  - Teacher logits -> soft target
  - Student logits -> log softmax
  - 使用 KLDivLoss（带温度系数 `T`，并乘 `T^2`）
- 数据清洗：使用预生成的 CleanLab 索引（`.npy`）筛选训练样本

### 1.3 推理阶段

- 加载最终模型（`dinov2_large`）
- 对测试图片目录批量预测
- 启用 TTA（水平翻转、垂直翻转、乘性扰动）
- 输出 `detection_results.csv`

---

## 2. 代码与目录说明

项目根目录（宿主机）建议结构如下：

```text
FGIC/
├─ Dockerfile
├─ build_docker.sh
├─ run_docker.sh
├─ README.md
├─ source/
│  ├─ entry.sh
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ configs/
│  │  ├─ webfg400_teacher_train.yaml
│  │  ├─ webfg400_model_train.yaml
│  │  ├─ webfg400_test.yaml
│  │  ├─ webinat5000_teacher_train.yaml
│  │  ├─ webinat5000_model_train.yaml
│  │  └─ webinat5000_test.yaml
│  └─ src/
│     ├─ train.py
│     ├─ detect.py
│     ├─ cleanlab/
│     ├─ data/
│     ├─ cache/
│     └─ models/
├─ train_output/
└─ test_output/
```

容器内固定工作目录：

- 项目目录：`/workspace`
- 数据目录：`/data`

---

## 3. 环境与硬件要求

### 3.1 推荐配置

- Linux + NVIDIA GPU（建议多卡）
- CUDA 驱动可用（`docker run --gpus all`）
- 显存建议 >= 32GB / 卡（不足时可调小 `batch_size`）
- 内存建议 >= 32GB

### 3.2 依赖来源

镜像基于：

- `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel`

Python 主要依赖（见 `source/requirements.txt`）：

- `torch`, `torchvision`
- `timm`（ConvNeXt）
- `transformers`（DINOv2）
- `cleanlab`
- `ttach`
- `tensorboard` 等

---

## 4. 数据集准备

项目需要两个数据集，均需准备训练集和测试集目录：

- WebFG-400
- WebiNat-5000

### 4.1 训练集目录格式（非常重要）

训练数据由 `MyDataset` 按类别目录读取，类别目录必须为零填充格式：

- WebFG-400（400类）：`000` ~ `399`（3位）
- WebiNat-5000（5000类）：`0000` ~ `4999`（4位）

示例：

```text
/data/webfg400/train/
├─ 000/
├─ 001/
├─ ...
└─ 399/

/data/webinat5000/train/
├─ 0000/
├─ 0001/
├─ ...
└─ 4999/
```

### 4.2 测试集目录格式

测试阶段按图片文件批量读取，目录下直接放图片即可：

```text
/data/webfg400/test/
├─ xxx1.jpg
├─ xxx2.png
└─ ...

/data/webinat5000/test/
├─ yyy1.jpg
└─ ...
```

### 4.3 数据路径配置

容器内默认数据配置：

- `source/src/data/data400.yaml`
- `source/src/data/data5000.yaml`

其中训练路径默认：

- `/data/webfg400/train`
- `/data/webinat5000/train`

测试路径默认：

- `/data/webfg400/test`
- `/data/webinat5000/test`

---

## 5. 预训练模型下载（ConvNeXt / DINO）

### 5.1 下载方式

项目默认 **自动下载** 预训练权重，无需手工下载：

- ConvNeXt 来自 `timm` 模型库
- DINOv2 来自 HuggingFace `transformers`

在 `source/main.py` 中已设置：

- `HF_ENDPOINT=https://hf-mirror.com`

用于加速 HuggingFace 下载。

### 5.2 缓存目录

下载后的模型会缓存到：

- `source/src/cache/ConvNeXt`
- `source/src/cache/DINOv2`

如果你需要重新验证下载来源，可删除对应缓存目录后重新运行任务。

### 5.3 参考模型来源

- ConvNeXtv2-large: `timm/convnextv2_large.fcmae_ft_in1k`
- DINOv2-large: `facebook/dinov2-large-imagenet1k-1-layer`

---

## 6. 构建与启动容器

### 6.1 构建镜像

在项目根目录执行：

```bash
docker build -t FGIC:latest .
```

或直接使用脚本：

```bash
bash build_docker.sh
```

### 6.2 标准运行模板

```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /path/to/webfg400_train:/data/webfg400/train \
  -v /path/to/webfg400_test:/data/webfg400/test \
  -v /path/to/webinat5000_train:/data/webinat5000/train \
  -v /path/to/webinat5000_test:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh <mode>
```

`<mode>` 支持：

- `train400_teacher`
- `train5000_teacher`
- `train400_main`
- `train5000_main`
- `test400`
- `test5000`

建议严格按照上述六步顺序执行，以确保教师模型权重正确生成并被主模型加载。

> 注意：入口脚本 `entry.sh` 实际执行的是 `python main.py --mode "$1"`。

---

## 7. 关键配置文件说明

### 7.1 训练配置

- `source/configs/webfg400_teacher_train.yaml`
- `source/configs/webfg400_model_train.yaml`
- `source/configs/webinat5000_teacher_train.yaml`
- `source/configs/webinat5000_model_train.yaml`

重点参数：

- `model_name`: 模型名（teacher: `convnextv2_large`, main: `dinov2_large`）
- `pretrained`: 是否加载预训练权重
- `batch_size`, `epoch`, `lr`
- `use_cleanlab`, `cleanlab`: 是否使用样本筛选索引
- `distill`: 是否蒸馏
- `teacher_name`, `teacher_weight`: 蒸馏教师模型与权重路径
- `distill_alpha`, `distill_temperature`: 蒸馏损失权重与温度

### 7.2 测试配置

- `source/configs/webfg400_test.yaml`
- `source/configs/webinat5000_test.yaml`

重点参数：

- `weights`: 最终模型权重路径（默认指向 `best_model.pt`）
- `model_name`: `dinov2_large`
- `batchsize`
- `data`
- `output_dir`

---

## 8. 输出文件说明

### 8.1 训练输出

目录：`train_output/<task_name>/`

常见内容：

- `best_model.pt`: 验证集最佳模型
- `last_model.pt`: 最后一个 epoch 权重
- `history.csv`: 训练/验证指标
- `training_curves.png`: 损失与准确率曲线

示例任务目录：

- `train_output/webfg400_teacher_convnextv2_large`
- `train_output/webfg400_main_dinov2_large`
- `train_output/webinat5000_teacher_convnextv2_large`
- `train_output/webinat5000_main_dinov2_large`

### 8.2 测试输出

目录：`test_output/<task>/`

文件：

- `detection_results.csv`

CSV 每行包含：

- `image_name, predicted_class`

---

## 9. 训练与蒸馏细节（实现对应）

### 9.1 Teacher 阶段

- `distill: False`
- 仅优化 ConvNeXt 模型
- 损失函数以 `GCE_loss` 为主

### 9.2 Main 阶段（蒸馏）

- `distill: True`
- Student: DINOv2
- Teacher: ConvNeXt（加载对应 `best_model.pt`，并冻结参数）
- 总损失：

```text
loss = alpha * distill_loss + (1 - alpha) * supervised_loss
```

其中：

- `distill_loss`: KLDiv(student/teacher, temperature=T) * T^2
- `supervised_loss`: 对真实标签的监督损失（当前实现使用 GCE）

---


## 10. 一键命令参考

### 10.1 构建

```bash
docker build -t FGIC:latest .
```

### 10.2 六步复现（按顺序）

```bash
# 1) WebFG-400 teacher
/bin/bash entry.sh train400_teacher

# 2) WebFG-400 main (distill)
/bin/bash entry.sh train400_main

# 3) WebFG-400 test
/bin/bash entry.sh test400

# 4) WebiNat-5000 teacher
/bin/bash entry.sh train5000_teacher

# 5) WebiNat-5000 main (distill)
/bin/bash entry.sh train5000_main

# 6) WebiNat-5000 test
/bin/bash entry.sh test5000
```

> 以上命令是在容器内部执行时的写法；宿主机执行请使用前文 `docker run ... /bin/bash entry.sh <mode>` 模板。

---

## 11. 许可与说明

- 本项目为 FGIC 任务工程化实现，重点在可复现训练流程与提交结果生成。
- 若用于比赛或论文复现，请自行核对数据与预训练模型使用许可。
