"""
Earth-Satallite-system-vispy.py — 卫星轨道 3D 可视化（UI 层）

基于 vispy（GPU 加速 3D 渲染）+ PyQt5（界面控件）。
数据层完全由 satellite_sim.py 提供，本文件只负责展示。

Controls:
  - 鼠标左键拖动：旋转
  - 鼠标中键/右键拖动：平移
  - 滚轮：缩放
  - 右侧面板：滑块 + 参数输入 + 控制按钮
"""

import sys
import time
import threading
import tempfile
import os
import subprocess
import numpy as np

import vispy.app
vispy.app.use_app('pyqt5')

from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene, visuals

from satellite_sim import SatelliteSimulator
from earth_globe import EarthGlobe
from continent_data import CONTINENTS
from Rotation import CalcQuiver3D


# ==================== 常量 ====================

RE = 6.371e6
DEFAULT_H_KM = 2000.0
DEFAULT_V0_KMS = 6.898174
DEFAULT_THETA_DEG = 90.0

VELOCITY_SCALE = 200.0
ACCEL_SCALE = 2_000_000.0
CAMERA_DISTANCE = 3.5e7
FPS_ALPHA = 0.05
SAVE_STRIDE = 4
SAVE_MAX_FRAMES = 600
EARTH_OMEGA = 7.2921e-5


# ==================== 可折叠面板 ====================

class CollapsibleSection(QtWidgets.QWidget):
    """点击标题展开/收起内容区。"""

    def __init__(self, title, collapsed=True, parent=None):
        super().__init__(parent)
        self._collapsed = collapsed
        self._btn = QtWidgets.QPushButton(("▸" if collapsed else "▾") + " " + title)
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setFlat(True)
        self._btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 6px; "
            "font-weight: bold; color: #ccc; border: none; }"
            "QPushButton:hover { color: #fff; background: rgba(255,255,255,0.05); }"
        )
        self._btn.toggled.connect(self._on_toggle)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(not collapsed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._btn)
        layout.addWidget(self._content)

    def _on_toggle(self, checked):
        self._content.setVisible(checked)
        self._btn.setText(("▾" if checked else "▸") + " " + self._btn.text()[2:])

    def content_widget(self):
        return self._content


# ==================== 主窗口 ====================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卫星轨道模拟 — 科里奥利力效应")
        self.resize(1400, 850)

        self._paused = True
        self._changed = True
        self._frame_idx = 0
        self._fps = 0.0
        self._last_frame_time = time.perf_counter()
        self._ui_lock = False       # 防止 slider ↔ spinbox 循环触发
        self._sim = SatelliteSimulator()
        self._result = None
        self._vel_scale = VELOCITY_SCALE
        self._acc_scale = ACCEL_SCALE
        self._play_speed = 1.0

        self._saving = False
        self._save_frames = []
        self._save_idx = 0
        self._save_total = 0

        self._setup_ui()
        self._setup_vispy()
        self._setup_timer()
        self._update_ref_info()
        self._update_preview()

    # ================================================================
    #  UI
    # ================================================================

    def _setup_ui(self):
        central = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(central)

        # -- 左侧：vispy 画布 --
        self._canvas_widget = QtWidgets.QWidget()
        canvas_layout = QtWidgets.QVBoxLayout(self._canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        hud_style = (
            "color: #e0e0e0; background: rgba(0,0,0,140); padding: 4px 8px; "
            "border-radius: 3px; font: 13px 'Consolas';"
        )
        self._lbl_time = QtWidgets.QLabel("Time: 0.0000 days", self._canvas_widget)
        self._lbl_time.setStyleSheet(hud_style); self._lbl_time.move(10, 10)
        self._lbl_error = QtWidgets.QLabel("Error: --", self._canvas_widget)
        self._lbl_error.setStyleSheet(hud_style); self._lbl_error.move(10, 40)
        self._lbl_fps = QtWidgets.QLabel("FPS: --", self._canvas_widget)
        self._lbl_fps.setStyleSheet(hud_style); self._lbl_fps.move(10, 70)

        central.addWidget(self._canvas_widget)

        # -- 右侧：控制面板 --
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(560)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(6, 8, 6, 8)

        # ≡ 参数（slider + spinbox 联动）
        grp_param = QtWidgets.QGroupBox("参数")
        param_layout = QtWidgets.QVBoxLayout(grp_param)
        param_layout.setSpacing(4)

        # H
        self._spin_H, self._slider_H = self._make_param_row(
            "H", 0, 100000, DEFAULT_H_KM, 1, " km",
            slider_max=10000, to_slider=lambda v: int(v / 10),
            to_value=lambda s: float(s * 10),
        )
        param_layout.addLayout(self._param_rows[-1][0])  # H row

        # v₀
        self._spin_v0, self._slider_v0 = self._make_param_row(
            "v₀", 0, 20, DEFAULT_V0_KMS, 4, " km/s",
            slider_max=20000, to_slider=lambda v: int(v * 1000),
            to_value=lambda s: s / 1000.0,
        )
        param_layout.addLayout(self._param_rows[-1][0])

        # v₀ 预设按钮（单独一行，放 v₀ slider 下方）
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addStretch()
        btn_circ = QtWidgets.QPushButton("圆轨道")
        btn_circ.clicked.connect(self._set_circular_speed)
        preset_row.addWidget(btn_circ)
        btn_esc = QtWidgets.QPushButton("逃逸")
        btn_esc.clicked.connect(self._set_escape_speed)
        preset_row.addWidget(btn_esc)
        param_layout.addLayout(preset_row)

        # θ
        self._spin_theta, self._slider_theta = self._make_param_row(
            "θ", 0, 180, DEFAULT_THETA_DEG, 2, "°",
            slider_max=1800, to_slider=lambda v: int(v * 10),
            to_value=lambda s: s / 10.0,
        )
        param_layout.addLayout(self._param_rows[-1][0])

        self._lbl_ref = QtWidgets.QLabel()
        self._lbl_ref.setStyleSheet("color: #999; font-size: 12px; padding: 2px 0;")
        param_layout.addWidget(self._lbl_ref)

        panel_layout.addWidget(grp_param)

        # ≡ 显示（可折叠）
        self._sect_display = CollapsibleSection("显示", collapsed=True)
        disp = self._sect_display.content_widget()
        disp_layout = QtWidgets.QVBoxLayout(disp)
        disp_layout.setContentsMargins(16, 4, 4, 4)
        disp_layout.setSpacing(4)

        disp_layout.addWidget(QtWidgets.QLabel("透明度:"))
        self._slider_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_opacity.setRange(5, 80)
        self._slider_opacity.setValue(15)
        self._slider_opacity.valueChanged.connect(self._on_opacity_changed)
        disp_layout.addWidget(self._slider_opacity)

        disp_layout.addWidget(QtWidgets.QLabel("播放速度:"))
        speed_row = QtWidgets.QHBoxLayout()
        self._slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_speed.setRange(5, 30)
        self._slider_speed.setValue(10)
        self._slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self._slider_speed, 1)
        self._lbl_speed = QtWidgets.QLabel("1.0×")
        self._lbl_speed.setFixedWidth(40)
        self._lbl_speed.setStyleSheet("color: #ccc;")
        speed_row.addWidget(self._lbl_speed)
        disp_layout.addLayout(speed_row)

        self._chk_texture = QtWidgets.QCheckBox("🌍 地球纹理")
        self._chk_texture.setChecked(True)
        self._chk_texture.toggled.connect(self._on_texture_toggled)
        disp_layout.addWidget(self._chk_texture)

        self._chk_rotation = QtWidgets.QCheckBox("🔄 地球自转")
        self._chk_rotation.setChecked(False); self._chk_texture.setChecked(True)
        self._chk_rotation.setToolTip("可视化地球绕轴自转（播放时锁定）")
        self._chk_rotation.toggled.connect(self._on_param_changed)  # 切换自转需重新运行
        disp_layout.addWidget(self._chk_rotation)

        panel_layout.addWidget(self._sect_display)

        # ≡ 说明（可折叠）
        self._sect_help = CollapsibleSection("说明", collapsed=True)
        hc = self._sect_help.content_widget()
        hl = QtWidgets.QVBoxLayout(hc)
        hl.setContentsMargins(16, 4, 4, 4)
        ht = QtWidgets.QLabel(
            "蓝色 = 惯性系轨迹 / 预览\n"
            "橙色 = 旋转系（科里奥利力）\n\n"
            "红箭头 = 惯性系速度\n"
            "绿箭头 = 旋转系速度\n"
            "紫箭头 = 加速度分量\n"
            "粉箭头 = 数值微分加速度\n\n"
            "拖动滑块可预览惯性系轨道"
        )
        ht.setWordWrap(True); ht.setStyleSheet("color: #aaa;")
        hl.addWidget(ht)
        panel_layout.addWidget(self._sect_help)

        panel_layout.addStretch()

        # 按钮
        self._btn_reset = QtWidgets.QPushButton("🔄 重置")
        self._btn_reset.setMinimumHeight(30)
        self._btn_reset.clicked.connect(self._reset)
        panel_layout.addWidget(self._btn_reset)

        self._btn_toggle = QtWidgets.QPushButton("▶ 开始")
        self._btn_toggle.setMinimumHeight(36)
        self._btn_toggle.setStyleSheet(self._green_style())
        self._btn_toggle.clicked.connect(self._toggle_animation)
        panel_layout.addWidget(self._btn_toggle)

        self._btn_save = QtWidgets.QPushButton("💾 保存")
        self._btn_save.clicked.connect(self._save_animation)
        panel_layout.addWidget(self._btn_save)

        central.addWidget(panel)

        self._statusbar = QtWidgets.QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("就绪 — 拖动滑块预览轨道")

    def _make_param_row(self, label, v_min, v_max, default, decimals, suffix,
                         slider_max, to_slider, to_value):
        """创建 slider + spinbox 联动行。"""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QtWidgets.QLabel(label + ":"))

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, slider_max)
        slider.setValue(to_slider(default))
        row.addWidget(slider, 1)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(v_min, v_max)
        spin.setDecimals(decimals)
        spin.setValue(default)
        spin.setSuffix(suffix)
        spin.setFixedWidth(110)
        row.addWidget(spin)

        # 双向同步
        def slider_changed(val, s=slider, sp=spin, tv=to_value):
            if self._ui_lock:
                return
            self._ui_lock = True
            sp.setValue(tv(val))
            self._ui_lock = False
            self._on_param_changed()

        def spin_changed(val, s=slider, sp=spin, ts=to_slider):
            if self._ui_lock:
                return
            self._ui_lock = True
            s.setValue(ts(val))
            self._ui_lock = False
            self._on_param_changed()

        slider.valueChanged.connect(slider_changed)
        spin.valueChanged.connect(spin_changed)

        if not hasattr(self, '_param_rows'):
            self._param_rows = []
        self._param_rows.append((row, slider, spin, to_slider, to_value))
        return spin, slider

    @staticmethod
    def _green_style():
        return (
            "QPushButton { background: #27ae60; color: white; font-size: 15px; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #2ecc71; }"
        )

    @staticmethod
    def _red_style():
        return (
            "QPushButton { background: #e74c3c; color: white; font-size: 15px; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #c0392b; }"
        )

    # ================================================================
    #  vispy 3D 场景
    # ================================================================

    def _setup_vispy(self):
        self._canvas = scene.SceneCanvas(
            keys='interactive', show=False, parent=self._canvas_widget,
        )
        self._canvas_widget.layout().addWidget(self._canvas.native)
        self._canvas.measure_fps()

        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.TurntableCamera(
            fov=45, elevation=30, azimuth=60,
            distance=CAMERA_DISTANCE, center=(0, 0, 0),
        )

        # 地球
        self._earth = EarthGlobe(
            radius=RE, rows=80, cols=160,
            face_color=(0.15, 0.35, 0.65, 0.15),
            wire_color=(0.5, 0.7, 1.0, 0.45),
        )
        self._earth.parent = self._view.scene
        self._earth.load_continents(CONTINENTS)

        # 坐标轴
        axis_len = 1.2e7
        for vec, color in [
            ((axis_len, 0, 0), '#e74c3c'),
            ((0, axis_len, 0), '#2ecc71'),
            ((0, 0, axis_len), '#3498db'),
        ]:
            ax = scene.visuals.Line(
                pos=np.array([[0, 0, 0], vec]),
                color=color, width=2, method='gl', parent=self._view.scene,
            )
            ax.set_gl_state(depth_test=False)

        empty = np.zeros((0, 3), dtype=np.float32)

        # 预览轨迹（半透明虚线色，惯性系）★
        self._trail_preview = scene.visuals.Line(
            pos=empty, color=(0.2, 0.6, 1.0, 0.5), width=1.5,
            method='gl', parent=self._view.scene,
        )
        self._trail_preview.set_gl_state(depth_test=False)

        # 实际轨迹
        self._trail_inertial = scene.visuals.Line(
            pos=empty, color='#3498db', width=2,
            method='gl', parent=self._view.scene,
        )
        self._trail_inertial.set_gl_state(depth_test=False)

        self._trail_rot = scene.visuals.Line(
            pos=empty, color='#e67e22', width=2,
            method='gl', parent=self._view.scene,
        )
        self._trail_rot.set_gl_state(depth_test=False)

        # 箭头
        self._arrows = {}
        for name, color in {
            'v_inertial': '#e74c3c', 'v_rot': '#2ecc71',
            'a_gravity': '#9b59b6', 'a_centrifugal': '#9b59b6',
            'a_coriolis': '#9b59b6', 'a_numerical': '#e91e90',
        }.items():
            arrow = scene.visuals.Line(
                pos=empty, color=color, width=1.5,
                method='gl', parent=self._view.scene, connect='segments',
            )
            arrow.set_gl_state(depth_test=False)
            self._arrows[name] = arrow

    # ================================================================
    #  动画
    # ================================================================

    def _setup_timer(self):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_timer_tick)

    def _on_timer_tick(self):
        if self._result is None or self._frame_idx >= len(self._result.t):
            return
        i = int(self._frame_idx)
        self._render_to_frame(i)

        # 地球自转（用模拟时间 dt）
        if self._chk_rotation.isChecked() and self._result is not None:
            if i > 0:
                sim_dt = self._result.t[i] - self._result.t[i - 1]
            else:
                sim_dt = self._result.t[1] - self._result.t[0]
            self._earth.rotate(np.degrees(EARTH_OMEGA * sim_dt), (0, 0, 1))

        # FPS
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps += FPS_ALPHA * (1.0 / dt - self._fps)

        # 播放速度：浮点步进，<1 时累积到 1 再进帧
        self._frame_idx += self._play_speed
        n = len(self._result.t)
        while self._frame_idx >= n:
            self._frame_idx -= n
        self._canvas.update()

    def _render_to_frame(self, i):
        r = self._result
        n = i + 1

        if self._chk_rotation.isChecked():
            # 自转 ON：橙=惯性，蓝=反方向自转（−ωt）
            self._trail_inertial.set_data(
                pos=np.column_stack((r.x_anti[:n], r.y_anti[:n], r.z_anti[:n]))
            )
            self._trail_rot.set_data(
                pos=np.column_stack((r.x_inertial[:n], r.y_inertial[:n], r.z_inertial[:n]))
            )
        else:
            # 默认：蓝=惯性，橙=旋转系（+ωt）
            self._trail_inertial.set_data(
                pos=np.column_stack((r.x_inertial[:n], r.y_inertial[:n], r.z_inertial[:n]))
            )
            self._trail_rot.set_data(
                pos=np.column_stack((r.x_rot[:n], r.y_rot[:n], r.z_rot[:n]))
            )

        if self._chk_rotation.isChecked():
            # 自转 ON：红箭头跟蓝色(反自转)，绿箭头跟橙色(惯性)
            p_blue = np.array([r.x_anti[i], r.y_anti[i], r.z_anti[i]])
            v_blue = np.array([r.vx_anti[i], r.vy_anti[i], r.vz_anti[i]])
            p_orange = np.array([r.x_inertial[i], r.y_inertial[i], r.z_inertial[i]])
            v_orange = np.array([r.vx_inertial[i], r.vy_inertial[i], r.vz_inertial[i]])
            acc_base = p_orange
        else:
            # 默认：红箭头跟蓝色(惯性)，绿箭头跟橙色(旋转系)
            p_blue = np.array([r.x_inertial[i], r.y_inertial[i], r.z_inertial[i]])
            v_blue = np.array([r.vx_inertial[i], r.vy_inertial[i], r.vz_inertial[i]])
            p_orange = np.array([r.x_rot[i], r.y_rot[i], r.z_rot[i]])
            v_orange = np.array([r.vx_rot[i], r.vy_rot[i], r.vz_rot[i]])
            acc_base = p_orange

        self._update_arrow('v_inertial',    p_blue,   v_blue,              self._vel_scale)
        self._update_arrow('v_rot',         p_orange, v_orange,            self._vel_scale)
        self._update_arrow('a_gravity',     acc_base, r.a_gravity[i],      self._acc_scale)
        self._update_arrow('a_centrifugal', acc_base, r.a_centrifugal[i],  self._acc_scale)
        self._update_arrow('a_coriolis',    acc_base, r.a_coriolis[i],     self._acc_scale)
        self._update_arrow('a_numerical',   acc_base, r.a_numerical[i],    self._acc_scale)

        days = r.t[i] / 86400.0
        err = np.linalg.norm(r.a_numerical[i] - r.a_theoretical[i])
        self._lbl_time.setText(f"Time: {days:.4f} days")
        self._lbl_error.setText(f"Error: {err:.4f} m/s²")
        self._lbl_fps.setText(f"FPS: {self._fps:.0f}")

    def _update_arrow(self, name, origin, vector, scale):
        # 与原版一致：CalcQuiver3D(pos+vec, vec) → 箭尾在 pos+vec，箭头在 pos
        scaled = vector * scale
        lines = CalcQuiver3D(*(origin + scaled), *scaled, arrow_length_ratio=0.3)
        if lines:
            self._arrows[name].set_data(pos=np.vstack(lines), connect='segments')

    # ================================================================
    # ================================================================
    #  重置
    # ================================================================

    def _reset(self):
        """重置所有参数到默认值，清空轨迹。"""
        if not self._paused:
            self._timer.stop()
            self._paused = True

        self._ui_lock = True
        self._spin_H.setValue(DEFAULT_H_KM)
        self._spin_v0.setValue(DEFAULT_V0_KMS)
        self._spin_theta.setValue(DEFAULT_THETA_DEG)
        self._ui_lock = False

        self._result = None
        self._frame_idx = 0
        self._changed = True
        empty = np.zeros((0, 3), dtype=np.float32)
        self._trail_inertial.set_data(pos=empty)
        self._trail_rot.set_data(pos=empty)
        for a in self._arrows.values():
            a.set_data(pos=empty)

        self._chk_rotation.setChecked(False); self._chk_texture.setChecked(True)
        self._slider_opacity.setValue(15)
        self._slider_speed.setValue(10)
        self._view.camera.distance = CAMERA_DISTANCE

        self._chk_rotation.setEnabled(True); self._chk_texture.setEnabled(True)
        self._slider_H.setEnabled(True)
        self._slider_v0.setEnabled(True)
        self._slider_theta.setEnabled(True)
        self._btn_toggle.setText("▶ 开始")
        self._btn_toggle.setStyleSheet(self._green_style())

        self._update_ref_info()
        self._update_preview()
        self._statusbar.showMessage("已重置")

    #  预览
    # ================================================================

    def _update_preview(self):
        """实时更新惯性系预览轨道 + 自动调整视野。"""
        try:
            H_m = self._spin_H.value() * 1000.0
            v0 = self._spin_v0.value() * 1000.0
            theta = np.radians(self._spin_theta.value())
            x, y, z = self._sim.preview_orbit(H_m, v0, theta)
            self._trail_preview.set_data(
                pos=np.column_stack((x, y, z))
            )
            # 自动调整相机距离以适应轨道大小
            r_max = np.max(np.sqrt(x**2 + y**2 + z**2))
            self._view.camera.distance = max(r_max * 2.8, RE * 3)
        except Exception:
            self._trail_preview.set_data(pos=np.zeros((0, 3), dtype=np.float32))

    def _fit_camera(self):
        """根据当前模拟结果调整视野。"""
        if self._result is None:
            return
        r = self._result
        r_inertial = np.sqrt(r.x_inertial**2 + r.y_inertial**2 + r.z_inertial**2)
        r_rot = np.sqrt(r.x_rot**2 + r.y_rot**2 + r.z_rot**2)
        r_max = max(r_inertial.max(), r_rot.max())
        self._view.camera.distance = max(r_max * 2.8, RE * 3)

    def _update_arrow_scales(self):
        """根据轨道参数自适应缩放箭头，避免不同轨道下箭头过大或过小。"""
        if self._result is None:
            return
        r = self._result
        r_max = max(
            np.max(np.sqrt(r.x_inertial**2 + r.y_inertial**2 + r.z_inertial**2)),
            RE * 1.5
        )
        v0 = self._spin_v0.value() * 1000.0
        a0 = SatelliteSimulator.GM / (self._spin_H.value() * 1000.0 + RE) ** 2
        # 箭头长度 ≈ 轨道的 25%（速度）/ 12%（加速度）当 v=v0, a≈a0
        self._vel_scale = r_max * 0.25 / max(v0, 1.0)
        self._acc_scale = r_max * 0.12 / max(a0, 1e-6)

    # ================================================================
    #  交互
    # ================================================================

    def _on_param_changed(self):
        """参数变更：标记需要重新计算 + 更新预览。"""
        if not self._changed:
            self._changed = True
            self._btn_toggle.setText("▶ 开始")
            self._btn_toggle.setStyleSheet(self._green_style())
        self._update_ref_info()
        self._update_preview()

    def _update_ref_info(self):
        H_m = self._spin_H.value() * 1000.0
        v_circ = SatelliteSimulator.circular_speed(H_m)
        period = SatelliteSimulator.orbital_period(H_m)
        self._lbl_ref.setText(
            f"圆轨道速度: {v_circ / 1000:.4f} km/s\n"
            f"周期: {period / 60:.1f} min"
        )

    def _on_texture_toggled(self, checked):
        self._earth.set_continents_visible(checked)

    def _set_circular_speed(self):
        H_m = self._spin_H.value() * 1000.0
        v_circ = SatelliteSimulator.circular_speed(H_m) / 1000.0
        self._ui_lock = True
        self._spin_v0.setValue(v_circ)
        self._ui_lock = False
        self._on_param_changed()

    def _set_escape_speed(self):
        H_m = self._spin_H.value() * 1000.0
        v_esc = np.sqrt(2) * SatelliteSimulator.circular_speed(H_m) / 1000.0
        self._ui_lock = True
        self._spin_v0.setValue(v_esc)
        self._ui_lock = False
        self._on_param_changed()

    def _on_speed_changed(self, value):
        self._play_speed = value / 10.0
        self._lbl_speed.setText(f"{self._play_speed:.1f}×")

    def _on_opacity_changed(self, value):
        self._earth.set_opacity(value / 100.0)

    def _toggle_animation(self):
        if self._paused:
            self._start_animation()
        else:
            self._pause_animation()

    def _start_animation(self):
        if self._changed:
            self._statusbar.showMessage("计算中...")
            QtWidgets.QApplication.processEvents()
            try:
                self._sim.set_params(
                    H=self._spin_H.value() * 1000.0,
                    v0=self._spin_v0.value() * 1000.0,
                    theta=np.radians(self._spin_theta.value()),
                )
                self._result = self._sim.run()
                self._frame_idx = 0
                self._changed = False
                self._fit_camera()
                self._update_arrow_scales()
            except Exception as e:
                self._statusbar.showMessage(f"计算出错: {e}")
                return

        # 播放时锁定预览和自转开关
        self._chk_rotation.setEnabled(False); self._chk_texture.setEnabled(False)
        self._slider_H.setEnabled(False)
        self._slider_v0.setEnabled(False)
        self._slider_theta.setEnabled(False)

        self._timer.start()
        self._paused = False
        self._last_frame_time = time.perf_counter()
        self._btn_toggle.setText("⏸ 暂停")
        self._btn_toggle.setStyleSheet(self._red_style())
        self._statusbar.showMessage("运行中...")

    def _pause_animation(self):
        self._timer.stop()
        self._paused = True
        # 暂停时恢复控件
        self._chk_rotation.setEnabled(True); self._chk_texture.setEnabled(True)
        self._slider_H.setEnabled(True)
        self._slider_v0.setEnabled(True)
        self._slider_theta.setEnabled(True)

        self._btn_toggle.setText("▶ 开始" if self._changed else "▶ 继续")
        self._btn_toggle.setStyleSheet(self._green_style())
        self._statusbar.showMessage("已暂停")

    # ================================================================
    #  保存
    # ================================================================

    def _save_animation(self):
        if self._result is None:
            self._statusbar.showMessage("请先运行模拟再保存")
            return

        self._statusbar.showMessage("正在保存动画...")
        self._btn_save.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._chk_rotation.setEnabled(False); self._chk_texture.setEnabled(False)  # 保存时锁定 ★

        was_playing = not self._paused
        if was_playing:
            self._timer.stop()
            self._paused = True

        self._saving = True
        self._save_frames = []
        self._save_idx = 0
        total_raw = len(self._result.t)
        step = max(1, total_raw // SAVE_MAX_FRAMES)
        self._save_total = min(total_raw, SAVE_MAX_FRAMES)
        self._save_step = step
        self._saved_frame_idx = self._frame_idx
        self._was_playing = was_playing

        self._save_timer = QtCore.QTimer()
        self._save_timer.timeout.connect(self._capture_save_frame)
        self._save_timer.start(0)

    def _capture_save_frame(self):
        if self._save_idx >= self._save_total:
            self._finish_save()
            return
        frame = self._save_idx * self._save_step
        if frame >= len(self._result.t):
            self._finish_save()
            return
        self._render_to_frame(frame)
        img = self._canvas.render()    # 渲染+回读，不调 update() 避免刷屏
        self._save_frames.append(img)
        self._save_idx += 1
        if self._save_idx % 100 == 0:
            self._statusbar.showMessage(f"保存中... {self._save_idx}/{self._save_total}")

    def _finish_save(self):
        self._save_timer.stop()
        self._saving = False
        self._btn_save.setEnabled(True)
        self._btn_toggle.setEnabled(True)
        self._frame_idx = self._saved_frame_idx
        self._statusbar.showMessage(
            f"已捕获 {len(self._save_frames)} 帧，后台合成视频中...")
        # ffmpeg 合成放后台线程（纯 CPU/IO，不碰 OpenGL）
        threading.Thread(target=self._frames_to_video, daemon=True).start()
        # 恢复暂停态控件
        self._chk_rotation.setEnabled(True); self._chk_texture.setEnabled(True)
        self._slider_H.setEnabled(True)
        self._slider_v0.setEnabled(True)
        self._slider_theta.setEnabled(True)

    def _frames_to_video(self):
        if not self._save_frames:
            return
        h, w = self._save_frames[0].shape[:2]
        output_path = os.path.abspath("Earth-Satallite system.mp4")
        fps = max(1, int(len(self._save_frames) / 10))
        try:
            proc = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{w}x{h}', '-pix_fmt', 'rgba', '-r', str(fps),
                '-i', 'pipe:0',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'fast', output_path,
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
               stdout=subprocess.DEVNULL)
            for img in self._save_frames:
                proc.stdin.write(img.tobytes())
            proc.stdin.close()
            proc.wait(timeout=120)
            self._statusbar.showMessage(
                f"已保存至 {output_path} ({len(self._save_frames)} 帧)")
        except FileNotFoundError:
            self._statusbar.showMessage("未找到 ffmpeg")
        except subprocess.TimeoutExpired:
            self._statusbar.showMessage("ffmpeg 超时")
            proc.kill()

    # ================================================================
    #  生命周期
    # ================================================================

    def closeEvent(self, event):
        self._timer.stop()
        if hasattr(self, '_save_timer'):
            self._save_timer.stop()
        self._canvas.close()
        super().closeEvent(event)


# ==================== 入口 ====================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QWidget { font-size: 16px; }")
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    p.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
    p.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    p.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
    p.setColor(QtGui.QPalette.Button, QtGui.QColor(55, 55, 55))
    p.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
    app.setPalette(p)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
