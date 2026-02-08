import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import scipy.integrate
from scipy.interpolate import interp1d
import threading
from matplotlib import animation
import matplotlib
from Rotation import Rotating, CalcQuiver3D
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False #正常显示负号
matplotlib.rcParams['animation.writer'] = 'ffmpeg'
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\Program Files (x86)\ffmpeg-2024-06-06-git-d55f5cba7b-full_build\bin\ffmpeg.exe'
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_frame_on(False)
ax1.tick_params(length = 0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('')
readme = '''
此页面会存在十秒。
说明：此程序运行时有黑色弹窗，切勿关闭，否则程序无法执行。
此程序用于研究地球自转下的卫星运动（科里奥利作用），用户可输入三个初始物理量：
H（单位：m）：距离地球表面高度（注：写入后会自动生成表现圆轨道的初速度）
v0（单位：m/s）：初速度（平行于地球表面方向）
theta（单位：rad）：初速度与赤道的夹角。
由于地球的对称性，所有轨道的初始位置被设置在x轴正半轴上，速度在yOz平面内。
用户点击start后展示动画，此期间不可更改参数，可点击pause后更改，重新生成轨迹
动画中，蓝色轨迹为不考虑地球自转的轨迹；橙色轨迹考虑地球自转的科里奥利力等。
红、绿箭头为蓝、橙色轨迹下的速度向量。对橙色轨道，另有粉、紫色向量表示加速度。
加速度向量中，紫色向量由速度微分求得，粉色向量由公式求得。
加速度向量与速度向量均有一定程度的放大。
time表示经过的时间，程序固定设置为1天，动画播放完重新播放。
Error表示不同的两种方法计算得合加速度的误差。
程序无反应时，可晃动鼠标刷新程序，或许有效。动画播放速度可能较慢。
'''
ax1.annotate(readme, (0, 0))
plt.pause(10)

H = 2e6
N = 5000  # 根据需要调整时间点的数量
#3.579e7
re = 6.371e6
changed = True
R = re + H
GM = 3.983324e14
vx = vy = vz = vx1 = vy1 = vz1 = x1 = y1 = z1 = x_uniform = y_uniform = z_uniform = vx_uniform = vy_uniform = vz_uniform = t_uniform = np.zeros((N,))
a_intern = a_calc = a_coriolis = a_centri = a_theore = np.zeros((N, 3))
x = np.array([R])
y = np.array([0.0])
z = np.array([0.0])
v0 = np.sqrt(GM / R)
theta = np.pi / 2
v = np.array([0.0, v0, 0.0])
v = Rotating(v, np.array([1, 0, 0]), theta)
t = np.array([])
#1,075.4589975481329539664395183784
omega = np.array([0., 0., 7.2921e-5])
paused = True

fig2 = plt.figure()
ax = fig2.add_axes([0.1, 0.15, 0.6, 0.7], projection='3d')

Rbox = fig2.add_axes([0.8, 0.8, 0.15, 0.05])
Rtext = TextBox(Rbox, 'H:', initial='2000000')
vbox = fig2.add_axes([0.8, 0.6, 0.15, 0.05])
vtext = TextBox(vbox, 'v0:', initial='6898.174')
abox = fig2.add_axes([0.8, 0.4, 0.15, 0.05])
atext = TextBox(abox, 'theta:', initial='1.57079633')
savebox = fig2.add_axes([0.8, 0.2, 0.15, 0.05])
savebut = Button(savebox, 'Save')
button_ax = fig2.add_axes([0.8, 0.1, 0.15, 0.05])
button = Button(button_ax, 'Start')

def submitR(text):
    global x, H, R, v0, GM, paused, changed
    try:
        H = (float)(text)
        R = re + H
        x = np.array([R])
        v0 = np.sqrt(GM / R)
        vtext.set_val(v0)
        paused = changed = True
        ani.event_source.stop()
    except ValueError:
        Rtext.set_val("ERROR!")
def submitv(text):
    global v0, v, paused, changed
    try:
        v0 = (float)(text)
        v = np.array([0.0, v0, 0.0])
        v = Rotating(v, np.array([1, 0, 0]), theta)
        paused = changed = True
        ani.event_source.stop()
    except ValueError:
        vtext.set_val("ERROR!")
def submita(text):
    global theta, v, paused, changed
    try:
        theta = (float)(text)
        v = np.array([0.0, v0, 0.0])
        v = Rotating(v, np.array([1, 0, 0]), theta)
        paused = changed = True
        ani.event_source.stop()
    except ValueError:
        atext.set_val("ERROR!")

Rtext.on_submit(submitR)
vtext.on_submit(submitv)
atext.on_submit(submita)


def draw_earth():
    global re
    u0 = np.linspace(0, 2*np.pi, 200)
    v0 = np.linspace(0, np.pi, 200)
    x0 = re * np.outer(np.cos(u0), np.sin(v0))
    y0 = re * np.outer(np.sin(u0), np.sin(v0))
    z0 = re * np.outer(np.ones(np.size(u0)), np.cos(v0))

    earth = ax.plot_surface(x0, y0, z0, alpha=0.1)
    return earth


def set_axes_equal(ax: plt.Axes):
    global x_uniform, y_uniform, z_uniform, x1, y1, z1
    x = np.concatenate((x_uniform, x1))
    y = np.concatenate((y_uniform, y1))
    z = np.concatenate((z_uniform, z1))
    xmean = np.mean(x)
    ymean = np.mean(y)
    zmean = np.mean(z)


    xrange = np.max(x) - np.min(x)
    yrange = np.max(y) - np.min(y)
    zrange = np.max(z) - np.min(z)
    # x = np.concatenate(x, np.array([-re, re]))
    # y = np.concatenate(y, np.array([-re, re]))
    # z = np.concatenate(z, np.array([-re, re]))
    lim = max(xrange, yrange, zrange) / 2.0


    if lim != 0 and xmean != None and ymean != None and zmean != None and lim != None:
        ax.set_xlim(xmean - lim, xmean + lim)
        ax.set_ylim(ymean - lim, ymean + lim)
        ax.set_zlim(zmean - lim, zmean + lim)
    else: 
        ax.set_xlim(-6.371e6, 6.371e6)
        ax.set_ylim(-6.371e6, 6.371e6)
        ax.set_zlim(-6.371e6, 6.371e6) 

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d(),])
    r = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x0, y0, z0 = r
    ax.set_xlim3d([x0 - radius, x0 + radius])
    ax.set_ylim3d([y0 - radius, y0 + radius])
    ax.set_zlim3d([z0 - radius, z0 + radius])
ax.set_box_aspect((1, 1, 1))

def dynamics(t, y):
    x1, y1, z1, vx1, vy1, vz1 = y
    r1 = np.array([x1, y1, z1])
    v1 = np.array([vx1, vy1, vz1])
    a1 = -GM/ np.linalg.norm(r1)**3 * r1
    dydt = np.concatenate((v1, a1))
    return dydt

t_bgn = 0
t_end = 86400
t_span = (t_bgn, t_end)

p, = ax.plot(x, y, z)  # 原始轨迹
p_g_o, = ax.plot(x, y, z, color='blue')  # 初始位置与原轨迹相同
earth = draw_earth()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, va='top')
a_text = ax.text2D(0.7, 0.95, '', transform=ax.transAxes, va='top')
ax.view_init(elev=30, azim=60)#, roll=90)

def trajectory_caculation():
    global x, y, z, v, t, vx, vy, vz, x1, y1, z1, vx1, vy1, vz1, x_uniform, y_uniform, z_uniform, vx_uniform, vy_uniform, vz_uniform, t_uniform, a_calc, a_centri, a_coriolis, a_theore, a_intern
    y0 = np.array([x[0], y[0], z[0]])
    y0 = np.concatenate((y0, v))
    solution = scipy.integrate.solve_ivp(dynamics, t_span, y0, method='Radau', atol=1e-6, rtol=1e-6)
    x = solution.y[0][:]
    y = solution.y[1][:]
    z = solution.y[2][:]
    vx = solution.y[3][:]
    vy = solution.y[4][:]
    vz = solution.y[5][:]
    t = solution.t
    t_uniform = np.linspace(t[0], t[-1], N)

    F_x = interp1d(t, x, kind='cubic', fill_value="extrapolate")
    F_y = interp1d(t, y, kind='cubic', fill_value="extrapolate")
    F_z = interp1d(t, z, kind='cubic', fill_value="extrapolate")
    F_vx = interp1d(t, vx, kind='cubic', fill_value="extrapolate")
    F_vy = interp1d(t, vy, kind='cubic', fill_value="extrapolate")
    F_vz = interp1d(t, vz, kind='cubic', fill_value="extrapolate")

    x_uniform = F_x(t_uniform)
    y_uniform = F_y(t_uniform)
    z_uniform = F_z(t_uniform)
    vx_uniform = F_vx(t_uniform)
    vy_uniform = F_vy(t_uniform)
    vz_uniform = F_vz(t_uniform)

    x1 = y1 = z1 = vx1 = vy1 = vz1 = np.array([])
    a_intern = np.zeros((N, 3))
    a_centri = np.zeros((N, 3))
    a_coriolis = np.zeros((N, 3))

    for i in range(0,len(t_uniform)):
        r1 = np.array([x_uniform[i], y_uniform[i], z_uniform[i]])
        v1 = np.array([vx_uniform[i], vy_uniform[i], vz_uniform[i]])
        theta = np.linalg.norm(omega) * t_uniform[i]
        r2 = Rotating(r1, np.array([0, 0, 1]), theta)
        v2 = Rotating(v1, np.array([0, 0, 1]), theta)
        a_intern[i] = r2 * (-GM / (np.linalg.norm(r2) ** 3))

        a_centri[i] = np.cross(omega, np.cross(omega, r2))
        a_coriolis[i] = np.cross(omega, v2)

        x1 = np.append(x1, r2[0])
        y1 = np.append(y1, r2[1])
        z1 = np.append(z1, r2[2])
        vx1 = np.append(vx1, v2[0])
        vy1 = np.append(vy1, v2[1])
        vz1 = np.append(vz1, v2[2])

    def calc_acceleration(v, t):
        dt_backward = dt_forward = np.diff(t)
        acceleration = np.zeros_like(v)
        acceleration[0] = (v[1] - v[0]) / dt_forward[0]
        for i in range(1, len(v) - 2):acceleration[i] = (v[i + 1] - v[i - 1]) / (2 * dt_forward[i])
        acceleration[-1] = (v[-1] - v[-2]) / dt_backward[-1]
        return acceleration

    ax_calc = calc_acceleration(vx1, t_uniform)
    ay_calc = calc_acceleration(vy1, t_uniform)
    az_calc = calc_acceleration(vz1, t_uniform)
    a_calc = np.array([ax_calc, ay_calc, az_calc]).T
    a_theore = a_intern + a_coriolis# + a_centri

vec = ax.quiver3D(
    x[0], y[0], z[0], vx[0], vy[0], vz[0], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='red',
    pivot='tail',
    linewidths=2)
vec1 = ax.quiver3D(
    x1[0], y1[0], z1[0], vx1[0], vy1[0], vz1[0], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='green',
    pivot='tail',
    linewidths=2)
vec_a_i = ax.quiver3D(
    x1[0], y1[0], z1[0], a_intern[0][0], a_intern[0][1], a_intern[0][2], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='violet',
    pivot='tail',
    linewidths=2)
vec_a_ce = ax.quiver3D(
    x1[0], y1[0], z1[0], a_centri[0][0], a_centri[0][1], a_centri[0][2], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='violet',
    pivot='tail',
    linewidths=2)
vec_a_co = ax.quiver3D(
    x1[0], y1[0], z1[0], a_coriolis[0][0], a_coriolis[0][1], a_coriolis[0][2], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='violet',
    pivot='tail',
    linewidths=2)
vec_a_ca = ax.quiver3D(
    x1[0], y1[0], z1[0], a_coriolis[0][0], a_coriolis[0][1], a_coriolis[0][2], 
    visible=True,
    normalize=False,
    arrow_length_ratio=0.5, 
    color='purple',
    pivot='tail',
    linewidths=2)

# print(earth.get_zorder(), p.get_zorder(), vec.get_zorder())

def update(i):
    global k, p, x1, y1, z1, vx, vy, vz
    p_g_o.set_data_3d(x_uniform[:i+1], y_uniform[:i+1], z_uniform[:i+1])
    p.set_data_3d(x1[:i+1], y1[:i+1], z1[:i+1])
    r = np.array([x_uniform[i], y_uniform[i], z_uniform[i]])
    v = np.array([vx_uniform[i], vy_uniform[i], vz_uniform[i]])
    r1 = np.array([x1[i], y1[i], z1[i]])
    v1 = np.array([vx1[i], vy1[i], vz1[i]])

    scale = 200
    v *= scale
    v1 *= scale
    
    # 构建 segments
    segments = CalcQuiver3D(*(r + v), *v, arrow_length_ratio=0.3)
    segments1 = CalcQuiver3D(*(r1 + v1), *v1, arrow_length_ratio=0.3)
    segmentsain = CalcQuiver3D(*(r1 + (a_intern[i] * 2000000)), *(a_intern[i] * 2000000), arrow_length_ratio=0.3)
    segmentsace = CalcQuiver3D(*(r1 + (a_centri[i] * 2000000)), *(a_centri[i] * 2000000), arrow_length_ratio=0.3)
    segmentsaco = CalcQuiver3D(*(r1 + (a_coriolis[i] * 2000000)), *(a_coriolis[i] * 2000000), arrow_length_ratio=0.3)
    segmentsaca = CalcQuiver3D(*(r1 + (a_calc[i] * 2000000)), *(a_calc[i] * 2000000), arrow_length_ratio=0.3)
    
    vec.set_segments(segments)  # 更新箭头的位置和方向
    vec1.set_segments(segments1)
    vec_a_i.set_segments(segmentsain)
    vec_a_ce.set_segments(segmentsace)
    vec_a_co.set_segments(segmentsaco)
    vec_a_ca.set_segments(segmentsaca)
    current_time = t_uniform[i]/86400
    time_text.set_text(f'Time: {current_time:.4f} days')
    data = a_calc[i][:] - a_theore[i][:]
    a_text.set_text(f'Error: {np.linalg.norm(data):.4f}')

    #ax.view_init(elev=30, azim=(t[i]%86400)/86400*360)

    #draw_earth()
    return (p, p_g_o,)  # 返回两条轨迹的更新结果

ani = animation.FuncAnimation(fig2, update, frames=(len(t_uniform)-1), interval = 1, repeat=True)
set_axes_equal(ax)

fig2.canvas.stop_event_loop()
plt.pause(0.001)
ani.event_source.stop()
ani.event_source._timer.stop()
def toggle_animation(event):
    global paused, ani, changed
    if paused:
        if changed:
            trajectory_caculation()
            ani = animation.FuncAnimation(fig2, update, frames=(len(t_uniform)-1), interval = 1, repeat=True)
            changed = False
        ani.event_source.start()
        Rtext.set_active(False)
        vtext.set_active(False)
        atext.set_active(False)    
        set_axes_equal(ax)
        fig2.canvas.draw_idle()
        paused = False
        button.label.set_text('Pause')
        button.color = '0.75'
    else:
        ani.event_source.stop()
        paused = True
        Rtext.set_active(True)
        vtext.set_active(True)
        atext.set_active(True)
        if not changed:button.label.set_text('Resume')
        else:button.label.set_text('Start')
        button.color = 'r'
        fig2.canvas.draw_idle()

button.on_clicked(toggle_animation)

def save_animation(event):
    thread = threading.Thread(target = lambda:ani.save('Earth-Satallite system.mp4', writer='ffmpeg', fps=60, extra_args=['-loglevel', 'verbose']))
    thread.start()    

savebut.on_clicked(save_animation)
plt.show()