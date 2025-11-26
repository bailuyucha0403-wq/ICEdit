# ICEdit Training Repository

This repository contains the training code for ICEdit, a model for image editing based on text instructions. It utilizes conditional generation to perform instructional image edits.

This codebase is based heavily on the [OminiControl](https://github.com/Yuanshi9815/OminiControl) repository. We thank the authors for their work and contributions to the field!

## Setup and Installation

```bash
# Create a new conda environment
conda create -n train python=3.10
conda activate train

# Install requirements
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code directory
  - `train/`: Training modules
    - `train.py`: Main training script
    - `data.py`: Dataset classes for handling different data formats
    - `model.py`: Model definition using Flux pipeline
    - `callbacks.py`: Training callbacks for logging and checkpointing
  - `flux/`: Flux model implementation
- `assets/`: Asset files
- `parquet/`: Parquet data files
- `requirements.txt`: Dependency list

## Datasets

Download training datasets (part of OmniEdit) to the `parquet/` directory. You can use the provided scripts `parquet/prepare.sh`.

```bash
cd parquet
bash prepare.sh
```

## Training

```bash
bash train/script/train.sh
```

You can modify the training configuration in `train/config/normal_lora.yaml`. 



## MoE-LoRA Training

```bash
bash train/script/train_moe.sh
```

You can modify the training configuration in `train/config/moe_lora.yaml`.

## Distributed training launchers（分布式启动器说明）

当前脚本同时提到 `accelerate launch` 和 Lightning 的 `Trainer`，它们可以结合使用且不会产生“双重分布式”冲突，原因如下：

- **共同点**：两种方式最终都依赖 PyTorch Distributed（DDP）的通信栈做梯度 AllReduce。无论是由 `accelerate` 还是 Lightning 来启动进程，真正的同步都是在 DDP 后端完成的。
- **角色划分**：在本仓库脚本中，`accelerate launch` 只负责拉起多进程并设置 `LOCAL_RANK`、`WORLD_SIZE` 等环境变量；训练循环仍由 Lightning 的 `Trainer.fit` 执行。
- **为何可组合**：`Trainer` 并未在脚本里显式指定 `devices`/`strategy` 时不会额外再 fork 进程，因此用 `accelerate launch` 先启动进程再交给 Lightning，不会出现重复的分布式初始化；若想启用梯度同步，只需在 `Trainer` 中设置 `strategy="ddp"` 等参数（每个进程 `devices=1`），即可复用同一套 DDP 环境。

### 这样结合的实际好处（对比单独使用）
- **继承各自长处**：`accelerate` 负责简化多机/多卡进程拉起与环境变量管理，Lightning 则聚焦训练循环、回调、日志、精度管理等高层能力；组合后同时保留两者的便利性。
- **快速迁移旧配置**：已有的 `accelerate_config.yaml` 或启停脚本可以复用，不必把启动逻辑全部改写成 Lightning CLI；只需在 `Trainer` 配置分布式策略即可获得梯度同步。
- **易于调试/切换**：遇到问题时可以单步运行单进程（关闭 `accelerate launch`），也可以改用纯 Lightning 的 `Trainer(accelerator="gpu", devices=n, strategy="ddp")` 方式；两条路径底层都走 DDP，切换成本低。

实践中，想要单独用 Lightning 自带的分布式也可以直接运行 `python -m src.train.train_moe` 并设置 `accelerator="gpu"`、`devices=<卡数>`、`strategy="ddp"`，底层同样走 DDP，同步方式一致。

### 如何确认两张卡的梯度确实同步了？
在设置了 `strategy="ddp"`（且每个进程 `devices=1`）后，可以用下面几项快速自查，确保两张 H800 的 Trainer 进程在同一套参数上做 AllReduce：

- **启动日志检查**：Lightning 日志会打印 `Using strategy DDP`、`GPU available: True` 等字样；`accelerate` 会输出 `LOCAL_RANK/MASTER_ADDR/MASTER_PORT`，这些说明已进入 DDP 模式并建立通信。
- **进程内校验**：在训练脚本里临时插入一行 `print(torch.distributed.get_world_size(), torch.distributed.get_rank())`（需在 `torch.distributed.is_initialized()` 为 True 后调用），两进程应分别输出 world size=2、rank 为 0/1。
- **参数对齐检查**：在完成一次 `optimizer.step()` 后，在所有进程上对同一参数张量做 `allreduce` 或直接打印前几个元素，应保持一致；若不同步会看到数值逐步漂移。
- **训练指标一致性**：开启 `strategy="ddp"` 后，Lightning 默认对 loss 做进程间平均并只在 rank 0 日志输出，训练/验证曲线应随两卡参与而加速收敛；若仍各跑各的，loss 曲线会在两份日志中独立出现且不做平均。

### 代码内置的同步自检
- 训练脚本默认开启 `DDPVerificationCallback`：在训练开始时做一次参数 checksum all-reduce 校验，在首次 backward 后做一次梯度 checksum 校验，并在 rank 0 打印结果。若不需要可在配置里设置 `enable_ddp_verification: false`，或用 `verify_ddp_grads_after_steps` 推迟梯度检查步数。
- 当 `WORLD_SIZE>1` 时，脚本会自动把 Trainer 的 `strategy` 缺省为 `ddp`，并将 `accelerator`/`devices` 设为 GPU 单卡 per-rank（也可在配置里显式覆盖）。这样在 `accelerate launch` 多卡模式下可以直接得到梯度同步并配合上面的自检输出。