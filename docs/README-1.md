# Docker镜像使用说明

## 大致训练思路

本模型使用DINOv2-large模型在ImageNet1k数据集上的预训练权重，通过模型蒸馏、数据清洗等方法进行在WebFG400、WebiNat5000数据集上完成训练。
由于需要使用模型蒸馏方法，故在进行正式模型训练之前，要先在WebFG400、WebiNat5000数据集上训练得到教师模型权重。我们使用ConvNeXtv2-large模型在ImageNet1k数据集上的预训练权重，训练得到教师模型权重。

本方法一共要训练以下模型：
- webfg400_teacher_convnextv2_large：WebFG400数据集上的教师模型
- webfg400_main_dinov2_large：WebFG400数据集上的最终模型
- webinat5000_teacher_convnextv2_large：WebiNat5000数据集上的教师模型
- webinat5000_main_dinov2_large：WebiNat5000数据集上的最终模型

教师模型必须先于使用其权重的最终模型完成训练。

## Docker镜像构成

工作目录为`/workspace`，数据目录为`/data`。

训练过程中产生的结果（如模型权重、训练损失曲线图等）会产生在`/workspace/train_output`下，子文件夹的名称取决于任务名称（例如webfg400_teacher_convnextv2_large、webinat5000_main_dinov2_large；完成所有教师模型、最终模型的训练后，该文件夹下应该有4个子文件夹）。测试结果会产生在`/workspace/test_output`下，子文件夹的名称同样取决于任务名称（完成测试后该目录下应该会有webfg400_results、webinat5000_results两个子文件夹，每个子文件夹里面有一个CSV结果文件）。

启动python脚本的参数设置在`/workspace/configs`下，每一个yaml文件对应一个任务的参数。一共有6个任务，其中4个为训练任务（见前文），2个为测试任务。

项目源码放置在`/workspace/src`下。其中，`/workspace/src/cleanlab`下存放数据清洗后WebFG400、WebiNat5000数据集保留图像的索引，文件类型为npy。`/workspace/src/data`下存放数据集软链接地址，任务进程中将根据该目录下yaml文件完成数据集加载。`/workspace/src/pretrained_cache`下为预训练参数缓存文件。

`/workspace/entry.sh`为入口脚本，运行该脚本，将会启动`/workspace/main.py`开始复现任务。入口脚本支持传参，传参格式见后文。

**要点提示：若需要验证预训练参数的来源合法性，请删除`/workspace/src/pretrained_cache`下文件！执行代码过程中，预训练参数会自动下载至该文件夹！**

## 预训练权重来源

- ConvNeXtv2-large预训练权重来源：[convnextv2_large.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_large.fcmae_ft_in1k#model-card-for-convnextv2_largefcmae_ft_in1k)
- DINOv2-large预训练权重来源：[dinov2-large-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-large-imagenet1k-1-layer)

## 路径挂载说明

### 数据路径挂载

为确保数据加载成功，使用4个`-v`选项将两个数据集的训练、测试数据分开挂载，示例如下：

```bash
docker run -d --name FGIC --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh train5000_teacher
```

上例中，`/mnt/7T/data/webfg400_train/train`应替换为具体类别训练图片文件夹的上级目录（具体类别指的是命名为000、001、002等的文件夹）；`/mnt/7T/data/webfg400_test_B/test_B`应替换为具体类别测试图片所在文件夹，即该路径下直接存放所有测试图片。

请确保数据集被挂载在正确的位置。

### 输出路径挂载

为方便获取输出文件，请按上例挂载输出文件夹。任务开始后，将会在主机中镜像同级目录下生成`train_output`和`test_output`两个文件夹，文件夹的内容与docker镜像内输出文件夹同步。*根据具体情况，可以选择不挂载该路径，并从docker内部获得输出文件，不会对结果造成影响。*

## 配置要求

内存建议不小于32GB，本项目仅支持多GPU，单个GPU显存不小于32GB。若GPU显存不足，可以尝试调小`/workspace/configs`目录下各yaml文件中`batchsize`值。修改源代码之后，必须执行以下命令重新build docker：

```bash
docker build -t FGIC:latest .
```

我们的配置：
- Ubuntu 18.04 64bit / Linux 4.15.0-213-generic
- 4 GPU Tesla V100-SXM2-32GB
- Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz
- Memory: 80GB

## 复现过程

脚本`entry.sh`可以通过以下参数指定任务类型：
- train400_teacher：训练400数据集的教师模型参数
- train5000_teacher：训练5000数据集的教师模型参数
- train400_main：训练400数据集的最终模型参数
- train5000_main：训练5000数据集的最终模型参数
- test400：测试400数据集的最终模型
- test5000：测试5000数据集的最终模型

按照以下顺序执行任务：
1. 训练400数据集的教师模型参数
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh train400_teacher
```

2. 训练400数据集的最终模型参数
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh train400_main
```

3. 测试400数据集的最终模型
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh test400
```

4. 训练5000数据集的教师模型参数
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh train5000_teacher
```

5. 训练5000数据集的最终模型参数
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh train5000_main
```

6. 测试5000数据集的最终模型
```bash
docker run --gpus all --shm-size=16g -it --rm \
  -u $(id -u):$(id -g) \
  -v /mnt/7T/data/webfg400_train/train:/data/webfg400/train \
  -v /mnt/7T/data/webfg400_test_B/test_B:/data/webfg400/test \
  -v /mnt/7T/data/webinat5000_train/train:/data/webinat5000/train \
  -v /mnt/7T/data/webinat5000_test_B/test_B:/data/webinat5000/test \
  -v $(pwd)/train_output:/workspace/train_output \
  -v $(pwd)/test_output:/workspace/test_output \
  FGIC:latest \
  /bin/bash entry.sh test5000
```

**注：请修改数据集路径为真实路径！请勿删除输出路径下的文件！**