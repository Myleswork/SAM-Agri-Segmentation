# 上游审计（2026-10-27）

- 工作区：`/root/train/sam_train`。SAM2-UNet：`https://github.com/Myleswork/SAM2-UNet.git`，main，`01598e5e9912ffb23f965ecbebf4d1dfecbaa56e`；SAM3-UNet：`https://github.com/Myleswork/SAM3-UNet.git`，main，`aca5dd7aa7ce0be93752635d5be6e65f8b0864c8`。两者审计时 Git 状态干净。仓库 README 指向 WZH0120 对应项目。原始源码未复制。
- SAM2：`SAM2UNet.py` 使用 `sam2_hiera_l.yaml` 构造完整 SAM2，再保留 Hiera trunk；冻结已有编码器参数，在 blocks 周围插入可训练 Adapter；RFB 和 U 形解码器返回主 logits 及两个辅助 logits，均为单通道。训练默认 352，三个输出各用结构损失（加权 BCEWithLogits + IoU），AdamW、余弦退火。
- SAM3：`SAM3UNet.py` 直接创建 patch 14、1024 维、32 层 ViT，冻结已有参数后插入 Adapter，轻量 U 形解码器返回单通道 logits；训练默认 336，结构损失、AdamW、余弦退火。两个项目头部都硬编码二值。
- 两个上游 loader 在编码器外使用 ImageNet 均值及标准差归一化；未发现该 UNet 编码器路径内部再次归一化。原 loader 将 mask bicubic 缩放，原测试脚本逐图 min-max：新框架不应复用这些行为。
- SAM2 官方要求 SAM2（非 SAM2.1）`sam2_hiera_large.pt`，`build_sam2` 加载 checkpoint 的 `model` 键并严格检查键；构造默认直接使用 CUDA。SAM3 要求 `sam3.pt`，按 `detector.backbone.vision_backbone.trunk.` 筛选键后 `strict=False`，因此适配时须额外严格检查缺失和意外键。不得混用 SAM3.1。缺少真实权重，无法验证格式和兼容性；未下载权重。SAM3 官方获取可能要求访问许可，须遵循上游官方指南。
- SAM2 requirements 指定 torch~=2.4.0、hydra-core 1.3.2；SAM3 仓库没有独立依赖声明，其模型文件依赖 timm，其他依赖需实际导入验证。WEEDSBL Python 3.11.18、torch 2.8.0+cu128、torchvision 0.23.0+cu128 缺若干上游依赖；新工程独立 .venv 已安装所需依赖，两模型随机权重短训练在 Python 3.11 可运行，版本组合并非官方完整训练环境验证。保留 WEEDSBL 环境不修改。
- 可通过独立上游源码路径调用模型，但 SAM2 Hydra 配置初始化、设备默认值和 SAM3 宽松权重加载需处理并测试。输入 stem、严格加载、数据注册、指标和实验管理需自行实现。两个项目默认尺寸不同，共同 448 未实测。
- WeedyRice 示例实际为 1280×960，RGB 为 uint8，mask 示例为 0/255，光谱示例为 RGB 存储的 uint8；仅示例不能证明全部文件。README 声称空间对齐，元数据通过原始文件名连接标准化文件名；需全量检查尺寸及波段，并且相同尺寸本身不能证明配准准确。

## 许可核对

本地 SAM2-UNet 的 `LICENSE` 和 SAM3-UNet 的 `LICENSE.txt` 均为 Apache-2.0；但 SAM3-UNet 内还包含 Meta 的 `sam3` 源码，不能仅凭包装仓库顶层许可证推断其权重和嵌入源码的全部授权。Meta 官方 [SAM3 License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) 对 SAM Materials 有单独条款，包括分发、研究发表致谢及用途限制；[官方 README](https://github.com/facebookresearch/sam3) 要求申请 checkpoint 访问权限，通过后身份验证下载。SAM2 官方 [LICENSE](https://github.com/facebookresearch/sam2/blob/main/LICENSE) 为 Apache-2.0。实际权重和数据许可、公司使用与再分发条件应由负责人员结合权重来源和适用条款审查；此记录不是授权结论。
