# 新增数据集

在 `datasets/adapters/` 添加模块，用 `@register('dataset_name')` 注册返回 PyTorch Dataset 的类，构造函数接收 `(split, config)`。`build_dataset` 自动发现该目录的模块，训练器不按数据集名称分支。新增 YAML 指定 `name`、输入模式和 `channels`、尺寸、归一化、数据划分与标签映射；请自行实现文件配对、同步变换及数据校验。

每个样本应包含 `image`（C×H×W float）、`mask`（1×H×W 二值）、`sample_id`、`dataset_id`、`split`、`input_spec` 和 `metadata`。如要通过当前数据集推理入口生成原始分辨率叠加图，还需 metadata 中的 `rgb_path`、`mask_path` 与原始标签前景值列表 `foreground_values`。当前训练、评价仅支持二值分割；ignore index 和多类标签尚未实现。测试应覆盖数据配对、通道、标签、增强同步、划分泄漏。
