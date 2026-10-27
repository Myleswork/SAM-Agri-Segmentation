# SAM-Agri-Segmentation（开发中）

这是一个尚未完成的农业图像分割实验框架。当前仅实现 WeedyRice 的只读完整性检查、数据加载原型、光谱投影层和二值混淆矩阵指标；目前不能训练、评价、预测或测速 SAM2/SAM3。不要将此目录当作已验证的实验平台。

## 当前运行环境

依赖下载使用 `http://127.0.0.1:17892`。当前 `.venv/pip.conf` 已设置 pip 代理；重建环境后可执行 `python -m pip config --site set global.proxy http://127.0.0.1:17892`。其他下载工具需设置 `HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY` 指向同一地址。

新建 `.venv` 使用 Python 3.11，并从 WEEDSBL 环境复制了已安装的 PyTorch/CUDA 等包；未修改原环境。其余依赖通过代理安装。该环境已通过现有单元测试，SAM2/SAM3 随机权重前向可运行；`pip check` 报告 decord 0.6.0 的 wheel 内部标记为 cp36，Python 3.11 兼容性尚未充分验证，不能视为已经完成的可复现环境。

```bash
cd /root/train/sam_train/SAM-Agri-Segmentation
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/inspect_dataset.py --output results/dataset_audit.json
```

数据检查会读取所有列入划分清单的 RGB、mask 和四波段，保存 JSON；遇到缺失、错误、波段通道不相等或重复划分时返回非零退出码。文件名按大小写不敏感匹配；空间尺寸一致不能证明图像像素已配准。前景比例以非零 mask 像素统计，并报告所有原始值。航次信息没有可靠的航次 ID；同日连续原始影像编号跨划分仅是潜在泄漏线索。

`datasets.registry.build_dataset('weedyrice', 'train', config)` 返回 PyTorch 数据集。配置示例见 `configs/datasets/weedyrice.yaml`；多光谱值除以显式的 `ms_scale`，这只是对存储值的数值缩放，并非反射率校准。加载时检查三通道 TIFF 数值完全相等，再取单通道；同步水平/垂直翻转可选，标签用最近邻插值。当前仅保留样本文件名和原尺寸，完整元数据接入尚未实现；忽略标签、多类别和一般数据集尚未实现。

`models.spectral_stem.SpectralStem` 在已归一化通道上投影到 RGB，初始化保留 RGB 通道、额外波段权重为零；独立张量测试已覆盖，尚未与上游模型联调。`metrics.segmentation.confusion_metrics` 对空分母定义为 1，暂不提供逐图或全局聚合。

上游结构、权重和依赖见 [审计](docs/upstream_audit.md)。下一步须完成模型工厂及严格权重加载、实验配置、checkpoint、训练、推理、聚合评价、测速与真实模型验证。缺少上游预训练权重，任何预训练推理速度或分割精度均未测量。
