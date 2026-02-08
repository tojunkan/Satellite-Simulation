import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.integrate import solve_ivp
from matplotlib import animation
x = np.array([3.633e8-4671000])
y = np.array([0.0])
z = np.array([0.0])
v = np.array([0.0, 1075.459, 0.0])

#1,075.4589975481329539664395183784
xe = np.array([4671000])
ye = np.array([0.0])
ze = np.array([0.0])
ve = np.array([0,0,0])

x_g_o = x
y_g_o = y
z_g_o = z
v_g_o = v
a0 = np.array([0., 0., 0.])
omega = np.array([0., 0., 7.2921e-5])
k = 1.0
GM = 3.983324e14
Gm = 4.897114e12
tmax = 1
tmin = 0
fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')

t_bgn = 0
t_end = 2.592e6

def draw_earth():
    re = 6.371e6
    u0 = np.linspace(0, 2*np.pi, 200)
    v0 = np.linspace(0, np.pi, 200)
    x0 = re * np.outer(np.cos(u0), np.sin(v0)) + xe
    y0 = re * np.outer(np.sin(u0), np.sin(v0)) + ye
    z0 = re * np.outer(np.ones(np.size(u0)), np.cos(v0)) + ze

    ax.plot_surface(x0, y0, z0)
draw_earth()

def set_axes_equal(ax: plt.Axes):
    global x, y, z
    xmean = np.mean(x)
    ymean = np.mean(y)
    zmean = np.mean(z)

    xrange = np.max(x) - np.min(x)
    yrange = np.max(y) - np.min(y)
    zrange = np.max(z) - np.min(z)
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
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x0, y0, z0 = origin
    ax.set_xlim3d([x0 - radius, x0 + radius])
    ax.set_ylim3d([y0 - radius, y0 + radius])
    ax.set_zlim3d([z0 - radius, z0 + radius])
ax.set_box_aspect((1, 1, 1))

p, = ax.plot(x, y, z)  # 原始轨迹
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, va='top')
p_g_o, = ax.plot(x, y, z, color='blue')  # 初始位置与原轨迹相同

def dynamics(t, y):

    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2= y

    r1 = np.array([x1, y1, z1])
    r2 = np.array([x2, y2, z2])

    r = r1 - r2

    v1 = np.array([vx1, vy1, vz1])
    v2 = np.array([vx2, vy2, vz2])

    a1 = -GM / np.linalg.norm(r)**3 * r1
    a2 = -Gm / np.linalg.norm(r)**3 * r2
    dydt = np.concatenate((v1, a1, v2, a2))
    
    return dydt

# 初始化条件示例
# 假设初始位置和速度分别为两个物体
y0 = np.array([x[-1], y[-1], z[-1]])
ye = np.array([xe[-1], ye[-1], ze[-1]])
y0 = np.concatenate((y0, v, ye, ve))

# 定义时间范围
t_span = (t_bgn, t_end)

# 调用solve_ivp解决合并的系统
solution = solve_ivp(dynamics, t_span, y0, method='RK45', atol=1e-9, rtol=1e-9)

x = solution.y[0][:]
y = solution.y[1][:]
z = solution.y[2][:]

t = solution.t


def update(i):
    global k, p, x, y, z, v, x_g_o, y_g_o, z_g_o, v_g_o
    p.set_data_3d(x[:i+1], y[:i+1], z[:i+1])
    current_time = t[i]/86400
    time_text.set_text(f'Time: {current_time:.2f} days')

    #ax.view_init(elev=30, azim=(t[i]%86400)/86400*360)
    

    set_axes_equal(ax)
    return (p, p_g_o,)  # 返回两条轨迹的更新结果

ani = animation.FuncAnimation(fig2, update, frames=len(x)-1)



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()