import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from scipy.integrate import solve_ivp
from matplotlib import animation
x = np.array([3.62e8])
y = np.array([0.0])
z = np.array([0.0])
v = np.array([0.0, -1022.0, 0.0])
a0 = np.array([0., 0., 0.])
omega = np.array([0., 0., 7.2921e-5])
k = 1.0
dt = 3600
GM = 3.98e14
tmax = 1
tmin = 0
fig2 = plt.figure()
# fig2.set_visible(False)
ax = fig2.add_axes([0.075, 0.1, 0.6, 0.8],projection='3d')

re = 6.371e6
u0 = np.linspace(0, 2*np.pi, 200)
v0 = np.linspace(0, np.pi, 200)
x0 = re * np.outer(np.cos(u0), np.sin(v0))
y0 = re * np.outer(np.sin(u0), np.sin(v0))
z0 = re * np.outer(np.ones(np.size(u0)), np.cos(v0))


ax.plot_surface(x0, y0, z0)

def set_axes_equal(ax: plt.Axes):
    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d(),])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

ax.set_box_aspect((1, 1, 1))

def set_lim():
    global tmax, tmin
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
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

p, = ax.plot(x, y, z)

# def f(v, r):
#     global x, y, z
#     R = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
#     V = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
#     a0 = r * (GM / (R**3))
#     Ep = GM / R
#     Ek = 0.5 * V ** 2
#     print(Ep + Ek)
#     return a0
#     # a = a0 + np.cross(omega, np.cross(omega, r)) + np.cross(2 * omega,  v) 
#     # return a

# def runge_kutta(v:np.ndarray, x:np.ndarray, t:float, f):
#     k1 = f(v, x)
#     k2 = f(v + 0.5 * t * k1, x + 0.5*t * v)
#     k3 = f(v + 0.5 * k2, x + 0.5*t * v + 0.25*t*t)
#     k4 = f(v + k3, x + t * v + 0.5*t*t)
#     return (v + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, x+t * v + t*t*(k1 + k2 + k3) / 6.0)

def acceleration(t, state):
    r, v_ = state[:3], state[3:]
    R = np.linalg.norm(r, ord=2)
    vdot = v_ - np.cross(omega, r)
    # V = np.linalg.norm(v_, ord=2)
    # Ep = -GM / R
    # Ek = 0.5 * V ** 2
    # print(Ep + Ek)
    if R == 0 or R == np.nan:raise Exception("Error! invalid R value!")
    a = -GM / (R**3) * r  - np.cross(omega, np.cross(omega, r)) - np.cross(2 * omega,  vdot) 
    return np.concatenate((v_, a))

def solve(x0, v0, dt):
    initial_state = np.concatenate((x0, v0))

    t_span = (0, dt)
    dt_min = 0.01
    dt_max = 10
    max_step = dt_max

    # 使用RK45求解器
    solution = solve_ivp(acceleration, t_span, initial_state, method='RK45', 
                        atol=1e-3, rtol=1e-3, max_step=max_step)
    return solution.y

def update(i):
    global k, dt, p, x, y, z, v
    r = np.array([x[-1], y[-1], z[-1]])
    r = solve(r, v, 3600)
    dx = r[0][-1]
    dy = r[1][-1]
    dz = r[2][-1]
    v = np.array([r[3][-1], r[4][-1], r[5][-1]])
    x = np.append(x, [dx])
    y = np.append(y, [dy])
    z = np.append(z, [dz])
    set_lim()
    set_axes_equal(ax)
    #ax.quiver(x[-1], y[-1], z[-1], v, color='blue')
    if i >= 1000:x, y, z = x[1:], y[1:], z[1:]
    p.set_data_3d(x, y, z)
    return (p,)

ani = animation.FuncAnimation(fig2, update, frames=None)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()