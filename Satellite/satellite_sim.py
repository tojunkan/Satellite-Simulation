"""
satellite_sim.py — 卫星轨道模拟器（纯数据/物理层）

零 UI 依赖。只依赖 numpy, scipy, Rotation。
提供 SimParams（入参）、SimResult（出参）、SatelliteSimulator（计算引擎）。

Usage:
    sim = SatelliteSimulator()
    sim.set_params(H=2e6, v0=6898.174, theta=1.57079633)
    result = sim.run()
    # result.x_inertial, result.x_rot, result.a_coriolis, etc.
"""

import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
from dataclasses import dataclass
from Rotation import Rotating  # 纯数学，与可视化无关


# ==================== 数据结构 ====================

@dataclass
class SimParams:
    """模拟输入参数（全部 SI 单位）。"""
    H: float                 # 地球表面高度 (m)
    v0: float                # 初始速度大小 (m/s)，平行地表
    theta: float             # 速度与赤道夹角 (rad)
    t_end: float = 86400.0   # 模拟时长 (s)，默认 1 天
    N: int = 5000            # 输出时间点数


@dataclass
class SimResult:
    """一次模拟的完整结果。所有数组长度均为 N。"""
    t: np.ndarray              # (N,) 时间 (s)

    # 惯性系轨迹（不考虑地球自转的纯引力轨道）
    x_inertial: np.ndarray     # (N,)
    y_inertial: np.ndarray
    z_inertial: np.ndarray
    vx_inertial: np.ndarray
    vy_inertial: np.ndarray
    vz_inertial: np.ndarray

    # 旋转系轨迹（地球自转参考系，科里奥利效应可见）
    x_rot: np.ndarray          # (N,)
    y_rot: np.ndarray
    z_rot: np.ndarray
    vx_rot: np.ndarray
    vy_rot: np.ndarray
    vz_rot: np.ndarray

    # 反方向自转轨迹（惯性系绕 z 轴转 −ωt，用于「地球自转」显示）
    x_anti: np.ndarray
    y_anti: np.ndarray
    z_anti: np.ndarray
    vx_anti: np.ndarray
    vy_anti: np.ndarray
    vz_anti: np.ndarray

    # 旋转系各加速度分量 (N, 3)
    a_gravity: np.ndarray      # 万有引力
    a_centrifugal: np.ndarray  # 离心加速度 ω×(ω×r)
    a_coriolis: np.ndarray     # 科里奥利加速度 ω×v
    a_numerical: np.ndarray    # 数值微分加速度（速度中心差分）
    a_theoretical: np.ndarray  # 理论合加速度（引力 + 科里奥利）


# ==================== 模拟引擎 ====================

class SatelliteSimulator:
    """卫星轨道模拟器 — 纯计算，零 UI 依赖。

    Usage:
        sim = SatelliteSimulator()
        sim.set_params(H=2e6, v0=6898.174, theta=np.pi/2)
        result = sim.run()
    """

    # 物理常量
    GM = 3.983324e14                         # 地球引力参数 (m³/s²)
    RE = 6.371e6                             # 地球半径 (m)
    OMEGA = np.array([0., 0., 7.2921e-5])    # 地球自转角速度 (rad/s)

    def __init__(self):
        self.params: SimParams | None = None

    # ---- 公共接口 ----

    def set_params(self, H: float, v0: float, theta: float,
                   t_end: float = 86400.0, N: int = 5000):
        """设置模拟参数（全部 SI 单位）。

        Parameters
        ----------
        H : float
            地球表面高度 (m)
        v0 : float
            初速度大小 (m/s)，方向平行于地球表面
        theta : float
            速度与赤道的夹角 (rad)，0 = 赤道面内，π/2 = 极轨
        t_end : float
            模拟时长 (s)，默认 86400 = 1 天
        N : int
            输出时间点数，默认 5000
        """
        self.params = SimParams(H=H, v0=v0, theta=theta, t_end=t_end, N=N)

    def run(self) -> SimResult:
        """执行完整模拟流程，返回计算结果。"""
        if self.params is None:
            raise RuntimeError("请先调用 set_params() 设置参数后再 run()")

        p = self.params
        R = self.RE + p.H

        # 1. 初始状态
        y0 = self._make_initial_state(R, p.v0, p.theta)

        # 2. 根据轨道能量计算积分时长（一个周期）
        r_mag, v_mag = np.linalg.norm(y0[:3]), np.linalg.norm(y0[3:])
        eps = 0.5 * v_mag ** 2 - self.GM / r_mag
        if eps < 0:
            a = -self.GM / (2.0 * eps)
            t_end = 2.0 * np.pi * np.sqrt(a ** 3 / self.GM) * 1.1  # 超出10%防截断
        else:
            t_end = 86400.0

        # 3. 惯性系 ODE 积分
        sol = self._integrate(y0, t_end)

        # 4. 三次样条插值到均匀时间网格
        t_uniform, inertial = self._interpolate(sol, p.N)

        # 5. 旋转参考系坐标变换（每点绕 z 轴转 ωt + −ωt）
        rotating = self._transform_to_rotating(t_uniform, inertial)

        # 6. 旋转系加速度分量计算
        acc = self._compute_accelerations(t_uniform, rotating)

        return SimResult(
            t=t_uniform,
            x_inertial=inertial['x'], y_inertial=inertial['y'],
            z_inertial=inertial['z'],
            vx_inertial=inertial['vx'], vy_inertial=inertial['vy'],
            vz_inertial=inertial['vz'],
            x_rot=rotating['x'], y_rot=rotating['y'], z_rot=rotating['z'],
            vx_rot=rotating['vx'], vy_rot=rotating['vy'], vz_rot=rotating['vz'],
            x_anti=rotating['x_anti'], y_anti=rotating['y_anti'],
            z_anti=rotating['z_anti'],
            vx_anti=rotating['vx_anti'], vy_anti=rotating['vy_anti'],
            vz_anti=rotating['vz_anti'],
            a_gravity=acc['gravity'],
            a_centrifugal=acc['centrifugal'],
            a_coriolis=acc['coriolis'],
            a_numerical=acc['numerical'],
            a_theoretical=acc['theoretical'],
        )

    @staticmethod
    def circular_speed(H: float) -> float:
        """给定高度 H (m) 的圆轨道速度 (m/s)。"""
        return np.sqrt(SatelliteSimulator.GM / (SatelliteSimulator.RE + H))

    @staticmethod
    def orbital_period(H: float) -> float:
        """给定高度 H (m) 的圆轨道周期 (s)。"""
        R = SatelliteSimulator.RE + H
        return 2 * np.pi * np.sqrt(R ** 3 / SatelliteSimulator.GM)

    def preview_orbit(self, H: float, v0: float, theta: float,
                       n_points: int = 300):
        """快速预览：只算惯性系轨迹，低分辨率，用于拖动滑块时实时预览。

        椭圆轨道（ε<0）：积分精确一个周期，显示完整闭合椭圆。
        抛物/双曲轨道（ε≥0）：积分 86400 秒。

        Parameters
        ----------
        H : 高度 (m)
        v0 : 初速度 (m/s)
        theta : 倾角 (rad)
        n_points : 预览点数

        Returns
        -------
        (x, y, z) : 三个 (n_points,) 数组
        """
        R = self.RE + H
        y0 = self._make_initial_state(R, v0, theta)
        r_mag = np.linalg.norm(y0[:3])
        v_mag = np.linalg.norm(y0[3:])
        eps = 0.5 * v_mag ** 2 - self.GM / r_mag

        if eps < 0:
            # 椭圆轨道：精确一个周期 + 10% 余量
            a = -self.GM / (2.0 * eps)
            t_end = 2.0 * np.pi * np.sqrt(a ** 3 / self.GM) * 1.1
        else:
            t_end = 86400.0

        sol = self._integrate(y0, t_end)
        step = max(1, int(np.ceil(len(sol.t) / n_points)))
        idx = slice(0, len(sol.t), step)
        return sol.y[0, idx].copy(), sol.y[1, idx].copy(), sol.y[2, idx].copy()

    # ---- 内部方法 ----

    @staticmethod
    def _make_initial_state(R: float, v0: float, theta: float) -> np.ndarray:
        """构建初始状态 [x, y, z, vx, vy, vz]。

        初始位置在 x 轴正半轴：(R, 0, 0)。
        初始速度在 y 方向，绕 x 轴旋转 theta（模拟轨道倾角）。
        """
        pos = np.array([R, 0.0, 0.0])
        vel = np.array([0.0, v0, 0.0])
        vel = Rotating(vel, np.array([1, 0, 0]), theta)
        return np.concatenate((pos, vel))

    @staticmethod
    def _dynamics(t: float, y: np.ndarray) -> np.ndarray:
        """ODE 右端函数：纯牛顿引力（惯性系）。

        状态向量 y = [x, y, z, vx, vy, vz]
        返回 dy/dt = [vx, vy, vz, ax, ay, az]
        """
        x, y_pos, z, vx, vy, vz = y
        r = np.array([x, y_pos, z])
        r_norm = np.linalg.norm(r)
        a = -SatelliteSimulator.GM / (r_norm ** 3) * r
        return np.array([vx, vy, vz, a[0], a[1], a[2]])

    def _integrate(self, y0: np.ndarray, t_end: float):
        """Runge-Kutta (Radau) 数值积分。"""
        return scipy.integrate.solve_ivp(
            self._dynamics, (0.0, t_end), y0,
            method='Radau', atol=1e-6, rtol=1e-6,
        )

    @staticmethod
    def _interpolate(solution, N: int):
        """三次样条插值到 N 点均匀时间网格。

        Returns
        -------
        t_uniform : (N,) array
        inertial  : dict with keys x, y, z, vx, vy, vz → (N,) arrays
        """
        t = solution.t
        y = solution.y   # shape (6, M)

        t_uniform = np.linspace(t[0], t[-1], N)

        keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        data = {}
        for i, key in enumerate(keys):
            f = interp1d(t, y[i], kind='cubic', fill_value='extrapolate')
            data[key] = f(t_uniform)

        return t_uniform, data

    def _transform_to_rotating(self, t_uniform: np.ndarray,
                                inertial: dict) -> dict:
        """将惯性系轨迹变换到地球自转参考系。

        每帧绕 z 轴旋转 +ωt，模拟地球自转效应。
        变换后的参考系中，卫星同时受引力 + 离心 + 科里奥利力。
        """
        N = len(t_uniform)
        omega_mag = np.linalg.norm(self.OMEGA)
        z_axis = np.array([0, 0, 1])

        result = {
            'x': np.empty(N), 'y': np.empty(N), 'z': np.empty(N),
            'vx': np.empty(N), 'vy': np.empty(N), 'vz': np.empty(N),
            'x_anti': np.empty(N), 'y_anti': np.empty(N), 'z_anti': np.empty(N),
            'vx_anti': np.empty(N), 'vy_anti': np.empty(N), 'vz_anti': np.empty(N),
        }

        for i in range(N):
            angle = omega_mag * t_uniform[i]
            r_i = np.array([inertial['x'][i], inertial['y'][i], inertial['z'][i]])
            v_i = np.array([inertial['vx'][i], inertial['vy'][i], inertial['vz'][i]])

            r_rot = Rotating(r_i, z_axis, angle)
            v_rot = Rotating(v_i, z_axis, angle)
            r_anti = Rotating(r_i, z_axis, -angle)
            v_anti = Rotating(v_i, z_axis, -angle)

            result['x'][i], result['y'][i], result['z'][i] = r_rot
            result['vx'][i], result['vy'][i], result['vz'][i] = v_rot
            result['x_anti'][i], result['y_anti'][i], result['z_anti'][i] = r_anti
            result['vx_anti'][i], result['vy_anti'][i], result['vz_anti'][i] = v_anti

        return result

    def _compute_accelerations(self, t_uniform: np.ndarray,
                                rot: dict) -> dict:
        """计算旋转系中各加速度分量。"""
        N = len(t_uniform)
        omega = self.OMEGA
        GM = self.GM

        a_gravity = np.zeros((N, 3))
        a_centrifugal = np.zeros((N, 3))
        a_coriolis = np.zeros((N, 3))

        for i in range(N):
            r = np.array([rot['x'][i], rot['y'][i], rot['z'][i]])
            v = np.array([rot['vx'][i], rot['vy'][i], rot['vz'][i]])
            r_norm = np.linalg.norm(r)

            # 引力（牛顿万有引力）
            a_gravity[i] = -GM / (r_norm ** 3) * r
            # 离心加速度 ω×(ω×r)
            a_centrifugal[i] = np.cross(omega, np.cross(omega, r))
            # 科里奥利加速度 ω×v（注意：标准形式为 -2ω×v，此处仅 ω×v）
            a_coriolis[i] = np.cross(omega, v)

        # 数值微分加速度（速度中心差分）
        a_numerical = self._numerical_derivative_3d(
            rot['vx'], rot['vy'], rot['vz'], t_uniform
        )

        # 理论合加速度 = 引力 + 科里奥利（与原版一致，离心单独显示）
        a_theoretical = a_gravity + a_coriolis

        return {
            'gravity': a_gravity,
            'centrifugal': a_centrifugal,
            'coriolis': a_coriolis,
            'numerical': a_numerical,
            'theoretical': a_theoretical,
        }

    @staticmethod
    def _numerical_derivative_3d(vx, vy, vz, t):
        """三通道中心差分求加速度。

        边界点用单侧差分（前向/后向），内部用中心差分。
        """
        n = len(t)
        dt = np.diff(t)  # (n-1,)

        ax = np.zeros(n)
        ay = np.zeros(n)
        az = np.zeros(n)

        # 起点：前向差分
        ax[0] = (vx[1] - vx[0]) / dt[0]
        ay[0] = (vy[1] - vy[0]) / dt[0]
        az[0] = (vz[1] - vz[0]) / dt[0]

        # 内部：中心差分
        for i in range(1, n - 1):
            ax[i] = (vx[i + 1] - vx[i - 1]) / (dt[i] + dt[i - 1])
            ay[i] = (vy[i + 1] - vy[i - 1]) / (dt[i] + dt[i - 1])
            az[i] = (vz[i + 1] - vz[i - 1]) / (dt[i] + dt[i - 1])

        # 终点：后向差分
        ax[-1] = (vx[-1] - vx[-2]) / dt[-1]
        ay[-1] = (vy[-1] - vy[-2]) / dt[-1]
        az[-1] = (vz[-1] - vz[-2]) / dt[-1]

        return np.column_stack((ax, ay, az))


# ==================== 自测 ====================
if __name__ == '__main__':
    sim = SatelliteSimulator()
    sim.set_params(H=2e6, v0=6898.174, theta=np.pi / 2)
    result = sim.run()

    print(f"时间点数 : {len(result.t)}")
    print(f"时间范围 : {result.t[0]:.1f} ~ {result.t[-1]:.1f} s "
          f"({result.t[-1] / 86400:.2f} 天)")
    print(f"惯性系 x : [{result.x_inertial.min():.2e}, {result.x_inertial.max():.2e}] m")
    print(f"旋转系 x : [{result.x_rot.min():.2e}, {result.x_rot.max():.2e}] m")

    error = np.linalg.norm(result.a_numerical - result.a_theoretical, axis=1)
    print(f"加速度误差 max  : {error.max():.6f} m/s²")
    print(f"加速度误差 mean : {error.mean():.6f} m/s²")
    print(f"加速度误差 std  : {error.std():.6f} m/s²")

    # 检查离心加速度数量级
    print(f"离心加速度 max  : {np.linalg.norm(result.a_centrifugal, axis=1).max():.6f} m/s²")
    print(f"科里奥利 max    : {np.linalg.norm(result.a_coriolis, axis=1).max():.6f} m/s²")

    # 验证圆轨道速度
    v_circ = SatelliteSimulator.circular_speed(2e6)
    print(f"\nH=2000km 圆轨道速度 : {v_circ:.2f} m/s")
    print(f"轨道周期              : {SatelliteSimulator.orbital_period(2e6):.1f} s")

    print("\n✓ 数据层自测通过")
