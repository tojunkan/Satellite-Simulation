# Satellite Simulation — 科里奥利力卫星轨道模拟

基于 vispy (GPU 加速 3D) + PyQt5 的交互式卫星轨道模拟器，研究地球自转参考系中科里奥利力对卫星轨道的影响。

![License](https://img.shields.io/badge/license-MIT-green)

## 功能

- **3D 实时渲染** — GPU 加速，惯性系（蓝色）和旋转系（橙色）双轨迹对比
- **交互式参数调节** — 滑块拖动 H/v₀/θ，实时预览惯性系椭圆轨道
- **科里奥利力可视化** — 旋转系中引力、离心力、科里奥利力加速度分量箭头
- **地球纹理** — 六大洲 Delaunay 三角网格，支持自转开关
- **播放速度调节** — 0.5× ~ 3.0× 变速播放
- **圆轨道/逃逸速度** — 一键预设初速度
- **动画录制** — 导出 MP4（需 ffmpeg）

## 安装

```bash
pip install vispy PyQt5 numpy scipy
```

Python 3.10+，Windows / macOS / Linux。

## 运行

```bash
cd Satellite
python Earth-Satallite-system-vispy.py
```

## 操作

| 操作 | 方式 |
|------|------|
| 旋转视角 | 鼠标左键拖动 |
| 平移视角 | 鼠标中键/右键拖动 |
| 缩放 | 滚轮 |
| 修改参数 | 拖动 H/v₀/θ 滑块 |
| 预设圆轨道速度 | 调好 H，点「圆轨道」按钮 |
| 开始/暂停 | 「▶ 开始」按钮 |
| 重置 | 「🔄 重置」按钮 |
| 保存动画 | 「💾 保存」按钮（需 ffmpeg） |

## 物理模型

在惯性系求解二体问题 ODE：

$$\ddot{\mathbf{r}} = -\frac{GM}{r^3}\mathbf{r}$$

再通过坐标旋转变换到地球自转参考系：

$$\mathbf{r}_{rot} = R_z(\omega t) \cdot \mathbf{r}_{inertial}$$

旋转系中卫星同时受引力 + 离心力 + 科里奥利力：

$$\mathbf{a}_{theoretical} = -\frac{GM}{r^3}\mathbf{r} + \boldsymbol{\omega}\times\mathbf{v}$$

与数值微分得到的加速度 $\mathbf{a}_{numerical}$ 对比验证。

## 架构

```
Satellite/
├── satellite_sim.py                    # 纯数据/物理层（零 UI 依赖）
├── Earth-Satallite-system-vispy.py     # UI 层（vispy + PyQt5）
├── earth_globe.py                      # 地球 3D 可视化组件
├── continent_data.py                   # 大陆三角网格数据
├── Rotation.py                         # 3D 旋转 + 箭头几何数学库
└── Earth-Satallite-system-Interactive.py  # matplotlib 原版（参考）
```

数据层 (`satellite_sim.py`) 不依赖任何可视化库，仅用 numpy + scipy，可独立运行和测试。

## 依赖

- [vispy](https://github.com/vispy/vispy) — GPU 加速 3D 渲染
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) — UI 框架
- [numpy](https://numpy.org/) / [scipy](https://scipy.org/) — 数值计算
- [ffmpeg](https://ffmpeg.org/) — 动画录制（可选）

## License

MIT
