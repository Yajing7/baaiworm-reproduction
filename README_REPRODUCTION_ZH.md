# BAAIWorm 论文复现说明

本文对应 Zhao et al., *Nature Computational Science* 4, 978-990 (2024)，DOI:
`10.1038/s43588-024-00738-w`。

## 已完成的可执行复现

在 `D:\Celegans\BAAIWorm` 目录运行：

```powershell
& "C:\Users\30767\anaconda3\python.exe" reproduce.py
```

检查完整仿真环境：

```powershell
& "C:\Users\30767\anaconda3\python.exe" check_full_reproduction.py
```

输出写入 `reproduction_output/`：

- `metrics.json`：数值指标；
- `correlation_reproduction.png`：65 个神经元的实验目标、优化前后相关矩阵；
- `pca_reproduction.png`：论文所述神经活动 PCA；
- `reservoir_reproduction.png`：80 个运动神经元到 96 块肌肉的线性 readout。
- `single_neuron_reproduction.png`：6 个代表神经元的实验/模型电生理轨迹和稳态响应；
- `connection_weight_reproduction.png`：优化后突触与缝隙连接权重分布。
- `body_kinematics_reproduction.png`：17 个身体采样点在 TBRCS 中的位置和速度波形。
- `figure6_perturbation_reproduction.png`：论文图 6 的七组结构/连接扰动对比。
- `figure4_source_reproduction.png`：用官方 Source Data 精确重绘图 4e-f 身体波形。
- `closed_vs_open_loop_reproduction.png`：图 5 闭环与补充图 8 开环动力学对比。

这个入口直接读取作者发布的训练结果和原始目标数据，不依赖 GUI、NEURON 或
CUDA，因此可在当前 Windows 环境确定性运行。它验证的是论文结果/已发布模型资产，
不是从随机初始化重新训练 136 神经元网络。

## 完整重训练与闭环仿真

作者官方测试环境是 Ubuntu 20.04、Python 3.8、CUDA 11.4、NEURON 8.0、
OptiX 7.0，推荐 NVIDIA RTX 3090。完整流程还需要编译 NMODL 机制、C++/CUDA
软体有限元与渲染模块，并运行 GPU 优化。作者训练脚本 `eworm_learn/run_eworm_v4.py`
默认 `ngpu = 8`，当前 Windows + RTX 4050 6 GB 不适合原样重跑。

建议在兼容的 Ubuntu 工作站执行：

1. 安装 Python 3.8、NEURON 8.0、PyTorch/CuPy（匹配 CUDA 11.4）。
2. 在 `eworm/components/mechanism` 运行 `nrnivmodl modfile`。
3. 在 `eworm_learn` 运行 `nrnivmodl components/mechanism/modfile`。
4. 修改 `run_eworm_v4.py` 的 `ngpu` 与显存相关参数后运行
   `./x86_64/special run_eworm_v4.py`。
5. 按官方 README 编译 `neuronXcore`，再运行开环或闭环 GUI 仿真。

注意：官方 `requirements.txt` 是 Ubuntu 系统环境的完整导出，含 `apturl`、
`python-apt`、`dbus-python` 等系统包，不能当作便携 Python requirements 直接安装。

## 复现判据

- 相关矩阵：用最终 65 条膜电位轨迹计算 Pearson 相关矩阵，与实验目标比较 MSE。
  论文报告 MSE 0.076；仓库当前提交所附 `v_final_eworm_v4.npy` 会得到接近但不完全
  相同的值，具体见 `metrics.json`。
- PCA：对每个神经元的膜电位去均值，对 65 神经元进行 SVD/PCA。
- 神经-肌肉 readout：以 100 ms 为窗口平均 80 个运动神经元膜电位，丢弃前 40
  个窗口，用岭回归（`alpha=1e-3`）拟合 96 块肌肉，与作者源码一致。
- 单神经元：复用作者发布的膜片钳数字化数据和 NEURON 输出，按原 notebook 的
  `4/7` 到 `5.9/7` 时间窗计算稳态 I-V 响应、RMSE 与相关系数。
- 连接结构：直接读取优化后的 136 神经元抽象电路，统计化学突触、缝隙连接、
  兴奋/抑制极性和权重分布，不需要加载 NMODL。
- 身体运动学：按作者 C++ `KeyWorm::LoadJsonStates` 的布局解析每帧数据：前 6 项
  为目标/坐标速度，随后是 `17×3` 相对位置和 `17×3` 相对速度。复现论文图 4e-f
  所示的头至尾波形，并比较头、中心和尾部速度。
- 图 6 扰动：读取论文 Zenodo Source Data 中的 control、移除神经突、打乱连接位置、
  打乱突触/缝隙连接权重、移除突触/缝隙连接七种条件，比较神经相关矩阵以及头、
  中心、尾部的相对位置和速度。默认数据路径为 `D:\Celegans\BAAIWorm_Source_Data\Source_Data\Figure 6`，也可用
  `--figure6-data` 指定。
- 闭环/开环：读取 Figure 5 和 Supplementary Figure 8 的输入、80 个运动神经元及
  96 块肌肉时间序列，并计算频谱集中度，量化论文所述周期输入产生规则周期运动、
  感觉反馈产生非周期活动的差别。

作者 `pre_interaction.py` 先在第 413 行把预测从膜电位变换为激活值，又在第 447 行
保存前重复执行一次相同变换。因此发布的 `video_offline_eworm.muscle-*.npy` 数值集中
在约 0.8。复现脚本会撤销第二次变换，再与重新计算结果比较；该差异属于发布代码的
序列化缩放问题，不是 reservoir 拟合失败。

## 官方 Source Data 的新增精确验证

- 补充图 1：按官方补充材料的 5 行×4 列结构重建 AWC(L)、AIY(L)、AVA(L)、RIM(L)、VD5。直接解析 HOC 文件绘制形态，并重绘实验/模型响应、稳态 I-V 和初始峰值 I-V。输出为 `supplementary1_electrophysiology_reproduction.png`。
- 补充图 3：直接读取 `Supplementary Figure 3/control.npy`，重算 65 个神经元的相关矩阵。MSE 为 `0.075006`，论文报告值为 `0.076`。
- 补充图 2：从 `syn_gj_dist.xlsx` 读取 6,565 个化学突触与 287 个缝隙连接的位置，复现经验分布及论文给出的逆高斯拟合参数。输出为 `supplementary2_connection_location_reproduction.png`。
- 补充图 4：按作者 `wave_detect.py` 的 301 点窗口、10 倍插值和汇总互相关算法复现肌肉波传播，最优相邻肌肉延迟为 0.17 s。输出为 `supplementary4_muscle_wave_delay_reproduction.png`。
- 补充图 6：汇总 41 个闭环仿真结果。三种打乱条件各包含 10 个种子，三种移除条件各包含 5 个结果；报告净位移、头/尾平均速度、尾头速度比及 95% t 置信区间。输出为 `supplementary6_multiseed_statistics.png`，逐种子数值保存在 `metrics.json`。
- 补充图 7：读取官方 `video_online_wout.pkl`，重绘 80 个运动神经元到 96 块肌肉的闭环 readout 权重矩阵和分布。输出为 `supplementary7_readout_weights_reproduction.png`。
- 补充图 9：读取 AVAL 的 369 个有效轴突区段与两条树突的膜电位轨迹，复现胞体去极化沿轴突的空间衰减。远端轴突只保留约 19.18% 的去极化幅度，输出为 `supplementary9_neurite_propagation_reproduction.png`。
- 图 4、图 5 与补充图 8：分别从官方位置/速度和闭环/开环时序数据重绘，并将全部定量结果写入 `reproduction_output/metrics.json`。

## 主文图 1–3 重建

运行 `python reproduce_main_figures.py`，生成 `main_figure1_overview_reproduction.png`、
`main_figure2_network_construction_reproduction.png`、`main_figure3_body_model_reproduction.png`
及 `main_figures_1_to_3_metrics.json`。图 2、3 的数值面板读取仓库发布的神经活动、HOC、
身体网格、肌肉激活及 17 点状态数据；图 1 是依论文图注重绘的系统示意。图 2 的离子通道
机制只做结构示意，图 3 未包含发布数据中不存在的渲染场景/力场面板，因此这些重建不应
视为像图 4/6 那样逐像素或逐数据点的官方 Source Data 重绘。

## 本地数据目录

完整官方数据保存在 `D:\Celegans\BAAIWorm_Source_Data\Source_Data`。
脚本按自身位置寻找此目录及其中的 `Figure 6` 子目录，项目整体移动后无需修改盘符。
可用 `--source-data-root` 和 `--figure6-data` 显式指定其他数据位置。
下载压缩包、重复图 6 子集、下载专用虚拟环境及临时缓存已在完整性校验后清理。
