# 第一阶段验收

基础 Git HEAD 为 `3ad2cc7450e7a0afc88f8dfb84f302eb58895857`；第一阶段修改在工作区，未推送。实际文件可用 `git status --short` 查看：新增 `engine/`、`models/factory.py`、六份 `configs/experiments/`、训练/评价/预测/测速及 smoke 工具、相关测试、环境快照与本报告；修改了数据集、注册、指标、忽略规则和 README。SAM2/SAM3 分别从相邻原始仓库薄封装调用，commit 为 `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e` / `aca5dd7aa7ce0be93752635d5be6e65f8b0864c8`。

WeedyRice 全量审计见 `results/dataset_audit.json`：438/148/148，全部配对，734 RGB 和 Mask，2936 波段，1280×960，标签 0/255，三个波段存储通道相同。325 对同日连续编号跨划分，存在空间泄漏风险；同尺寸不证明配准。

在物理 GPU 2 的随机权重工厂测试中，SAM2 352、SAM3 336、双方 448 单图前向形状正确且有限。`results/smoke_sam2.json`、`results/smoke_sam3.json` 记录 RGB/4/7 通道前向、随机和真实 7 通道反向、AdamW 更新、SpectralStem 非 RGB 权重更新及 checkpoint 恢复输出最大误差 0。`results/smoke_modalities.json` 补充真实 RGB 和 4 通道反向，`results/smoke_bf16.json` 记录真实七通道 BF16 反向与更新。模型分别通过统一训练器完成 3 Batch 训练、2 Batch 验证及 2 Batch 测试；还分别完成 1 Batch BF16 训练闭环。所有这些都是随机初始化工程测试，不是精度结果。SAM2 额外从第一 epoch checkpoint 恢复并执行第二 epoch；各模型完整 checkpoint 的预测恢复一致性已测试。大型 best/last 文件仅保存在 git 忽略的 `results/runs/`。

独立评价与训练末尾评价的两图混淆矩阵完全一致；两个模型的数据集模式各生成一张 1280×960 预测和 5120×960 四列对比图。RGB 单图输入也已测试。指标测试覆盖空类、FP/FN、Micro/Macro 和前景覆盖率分组；独立模拟数据集通过注册后直接进入训练器；未定义指标不强制赋满分。分组结果不能替代整体指标。

## RTX 5090 随机初始化结构测速

48 项记录见 `results/benchmark_random_fp32strict/benchmark.{json,csv,md}`：每模型原生尺寸和 448，共 3 个 batch、两种精度及两种计时范围，各 30 次预热与 100 次计时，均成功。Eager，TF32 关闭，CUDA Event 纯前向；端到端包含重复读取真实 RGB 文件、预处理、H2D、前向、阈值和 D2H，不含保存。以下为 Batch=1、FP32：

| 模型 | 尺寸 | 前向平均 ms | P95 ms | 图像/s | 前向峰值 MiB | 端到端平均 ms |
|---|---:|---:|---:|---:|---:|---:|
| SAM2 | 352 | 19.47 | 19.59 | 51.37 | 899 | 37.72 |
| SAM3 | 336 | 19.26 | 19.29 | 51.93 | 1814 | 35.93 |
| SAM2 | 448 | 23.38 | 23.40 | 42.77 | 938 | 43.00 |
| SAM3 | 448 | 34.30 | 34.31 | 29.16 | 1897 | 53.32 |

原生尺寸不同，不能直接作为统一尺寸比较；这是计算成本，不能代表分割质量。初测 `results/benchmark_random/` 因 TF32 未统一而中断，不用于比较。正式基准启动于脏工作区；JSON 记录基础 commit 和测量时的源码哈希，精确的 factory 与 benchmark 源码快照保存在结果目录。随后 factory 增加的变更仅防止构建时写上游 bytecode，不改变前向计算。

本机未发现 SAM2 Hiera-L 或 SAM3 预训练权重；严格加载逻辑已实现但真实权重兼容性尚未验证。`sam2_hiera_large.pt` 参见 SAM2-UNet README，`sam3.pt` 依官方渠道和许可获取。不能混用 SAM2.1/SAM3.1。对实际文件还需检查来源、键与形状，不能仅依靠显式版本标记确认来源。

Python 3.11、torch 2.8.0+cu128；`decord` wheel 的内部 cp36 标签触发 pip check 报告不兼容，虽然静态图训练与导入已通过，视频路径未验证。共享内存 64 MB，num_workers 默认 0。版本记录见 `requirements-lock.txt`；安装走 17892 代理。上游目录无 Git 变更；WEEDSBL 保留原状。

## 遗留与正式实验前操作

多类/ignore index、滑窗、空间独立测试集和真实预训练权重加载仍未完成。推理单图目录模式限 RGB；多光谱需数据集提供真实波段。预训练模型和未训练分割头的直接推理接口尚无真实权重验证。正式实验前需取得相应权重和授权，填入配置并完成严格加载验证；复核空间划分，制定相同尺寸、预算及多种子协议，再显式扩大 smoke 配置的 epoch 和批次数。不要将 smoke 的两张测试图指标当成实验结论。

最终 CPU 单元测试：`.venv/bin/python -m unittest discover -s tests -v`，28 项通过；GPU 短测试独立运行，不能把这些结果视为真实预训练权重集成测试。`pip check` 返回非零并报告 `decord 0.6.0 is not supported on this platform`。本次未进行完整训练、空间独立测试、预训练质量评价或远程推送。
