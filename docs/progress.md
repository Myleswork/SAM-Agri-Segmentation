# 当前进度

第一阶段数据适配、随机权重 SAM2/SAM3 模型工厂、三种输入模式、训练/验证/最佳 checkpoint/恢复/测试、独立评价与推理、随机权重 RTX 5090 测速已实现并进行了短流程检查。GPU 2 上两个模型均进行了真实数据的 RGB/4/7 通道前向反向、7 通道 BF16 训练及 SpectralStem 更新；完整 checkpoint 加载后输出一致。详见 [第一阶段验收](phase1_acceptance.md)。

关键限制仍是无 SAM2/SAM3 预训练权重，真实预训练加载与预训练质量未验证；`decord` 的 wheel 元数据与 Python 3.11 的平台标签不一致；WeedyRice 官方划分有跨集合相邻航拍影像风险。禁止将随机权重 smoke 的小样本结果用于准确率结论。
