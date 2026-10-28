# SAM-Agri-Segmentation

统一数据集、SAM2-UNet/SAM3-UNet 二值分割、RGB/多光谱早期融合、训练、评价、推理与 RTX 5090 测速的第一阶段实验工程。WeedyRice 是首个适配器。当前没有已验证的预训练权重；内置配置的随机初始化仅用于短流程测试，不能用于精度结论。

## 环境与权重

上游源码必须位于本目录的相邻目录 `../SAM2-UNet`、`../SAM3-UNet`，版本和接口见 [上游审计](docs/upstream_audit.md)。使用 Python 3.11，独立虚拟环境；以下为重新安装方案，下载均走 17892 代理：

```bash
python3.11 -m venv .venv
export HTTP_PROXY=http://127.0.0.1:17892 HTTPS_PROXY=http://127.0.0.1:17892 ALL_PROXY=http://127.0.0.1:17892
.venv/bin/python -m pip config --site set global.proxy http://127.0.0.1:17892
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`requirements-lock.txt` 是当前环境的完整版本快照，含从 WEEDSBL 复制来的非必需包；先安装 cu128 的 torch/torchvision，再按需使用该快照。现有 `.venv` 通过复制既有 PyTorch 包并安装额外依赖构建；`decord 0.6.0` wheel 内部标记 cp36，Python 3.11 能导入，但 pip check 报平台不支持，视频路径兼容性未验证。静态图片上游模型及短训练已测试。共享内存 64 MB，默认 `num_workers: 0`。

上游训练需要 SAM2 `sam2_hiera_large.pt`（SAM2，非 SAM2.1），参考 [上游下载说明](https://github.com/WZH0120/SAM2-UNet#training)；SAM3 需要 `sam3.pt`，按 [官方 SAM3 指南](https://github.com/facebookresearch/sam3) 获得授权。建议分别放在 `checkpoints/pretrained/sam2_hiera_large.pt` 与 `checkpoints/pretrained/sam3.pt`，在实验配置填写 `pretrained` 和 `pretrained_version: sam2` 或 `sam3`。本机未发现这些文件；未下载，无法宣称已通过真实权重验证。代码许可证请分别查看上游仓库，权重、数据和业务部署的具体许可需另行核对。

## 数据与运行

数据位于 `/root/train/dataset/weedy-rice/extracted/WeedyRice-RGBMS-DB`。数据检查、样本和新增适配器见 [新增数据集](docs/add_dataset.md)；数据集未覆盖。七通道顺序明确区分 RGB 与多光谱的 G/R。多光谱 `ms_scale: 255` 只是存储值缩放，不是物理反射率。数据集的尺寸相同不能证明配准；官方划分中存在跨集合相邻航拍影像，不能声称空间独立。

```bash
cd /root/train/sam_train/SAM-Agri-Segmentation
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/inspect_dataset.py --output results/dataset_audit.json
export CUDA_VISIBLE_DEVICES=2
.venv/bin/python -u tools/train.py --config configs/experiments/sam2_unet_rgbms7.yaml --output results/runs/s2_example
.venv/bin/python tools/evaluate.py --checkpoint results/runs/s2_example/best.pth --split test --output results/runs/s2_eval
.venv/bin/python tools/predict.py --checkpoint results/runs/s2_example/best.pth --split test --limit 3 --output results/predictions/s2
.venv/bin/python tools/benchmark.py --image /path/to/image.JPG --output results/benchmark_random
```

每个模型有 RGB、RGB+NIR、RGB+G/R/RE/NIR 三份配置；默认每次只训练 3 Batch、验证和测试各 2 Batch。正式训练必须显式修改 epoch、批次数和权重路径，不要把 smoke 配置的指标当成模型精度。Checkpoint 恢复使用 PyTorch pickle 格式，仅加载可信来源的文件。训练目录必须为空；`--resume /path/to/last.pth` 可继续相同配置并恢复随机状态、优化器和调度器。参数冻结、BF16、梯度累积、阈值等在配置中控制。阈值调优只能用验证集。

评价对像素全局汇总为 Micro，逐图均值为 Macro；没有真实或预测前景时前景 IoU、Dice 为未定义，Macro 排除并记录有效样本数；Precision/Recall 在各自分母为零时同样未定义。mIoU 仅平均存在并可定义的类别。分组区间左闭右开，最后一组包含 100%。JSON 中的未定义值保存为 `null`，CSV 中为 `nan`。

预测可以使用 `--split test`，七通道通过 Dataset 读取实际波段；单图或目录使用 `--input`，仅限 RGB checkpoint。带有效 `pretrained` 路径的配置可用 `--pretrained-config` 进行预训练编码器加随机分割头的接口推理，该路径尚无真实权重验证。输出二值图、`.npy` 概率图和 RGB；有标签时另外输出 GT 和 TP/FP/FN 彩色叠加。整图缩放到模型尺寸，预测 logits 插值回原尺寸；滑窗尚未实现。

测速区分 CUDA Event 前向与读取 RGB、预处理、传输、前向、后处理和回传的端到端耗时，端到端不含保存文件。见 [实验规范](docs/experiment_protocol.md) 与 [验收记录](docs/phase1_acceptance.md)。不要将随机权重测速当成分割准确率。

断点恢复的最小复现（每个 epoch 一批，不进行正式训练）：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u tools/train.py --config configs/experiments/resume_smoke.yaml --output results/runs/resume_demo --stop-after-epoch 1
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u tools/train.py --config configs/experiments/resume_smoke.yaml --output results/runs/resume_demo --resume results/runs/resume_demo/last.pth
```
