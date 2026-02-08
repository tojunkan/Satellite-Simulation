import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import time
from scipy.integrate import solve_ivp
from matplotlib import animation
x = np.array([0.0, 3.62e8, 3.62e8])
y = np.array([0.0, 0.0, -1022.0])
z = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, -1022.0, 0.0])
GM = 3.98e14
tmax = 1
tmin = 0
fig2 = plt.figure()
# fig2.set_visible(False)
ax = fig2.add_axes([0.075, 0.1, 0.6, 0.8],projection='3d')


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

ax.set_box_aspect((1, 1, 1))

def set_lim():
    global tmax, tmin

    tmax = max(np.max(x), np.max(y), np.max(z))

    tmin = min(np.min(x), np.min(y), np.min(z))

    if tmax != tmin:
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(tmin, tmax)
        ax.set_zlim(tmin, tmax)
    else: 
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

def draw():
    ax.clear()
    ax.scatter(x[0:2], y[0:2], z[0:2], color='red')
    ax.quiver(x[1], y[1], z[1], x[2] - x[1], y[2] - y[1], z[2] - z[1], color='blue')
    set_lim()
    set_axes_equal(ax)

draw()

xbox = fig2.add_axes([0.8, 0.8, 0.15, 0.05])
xtext = TextBox(xbox, 'X:', initial='1.0')
ybox = fig2.add_axes([0.8, 0.7, 0.15, 0.05])
ytext = TextBox(ybox, 'Y:', initial='1.0')
zbox = fig2.add_axes([0.8, 0.6, 0.15, 0.05])
ztext = TextBox(zbox, 'Z:', initial='1.0')

vxbox = fig2.add_axes([0.8, 0.5, 0.15, 0.05])
vxtext = TextBox(vxbox, 'vX:', initial='0.0')
vybox = fig2.add_axes([0.8, 0.4, 0.15, 0.05])
vytext = TextBox(vybox, 'vY:', initial='0.0')
vzbox = fig2.add_axes([0.8, 0.3, 0.15, 0.05])
vztext = TextBox(vzbox, 'vZ:', initial='0.0')

Startbtnbox = fig2.add_axes([0.8, 0.2, 0.15, 0.05])
Startbtn = Button(Startbtnbox, 'Start!')
Stopbtnbox = fig2.add_axes([0.8, 0.1, 0.15, 0.05])
Stopbtn = Button(Stopbtnbox, 'Stop!')

def submitx(text):
    global x
    try:
        x[2] -= x[1]
        x[1] = (float)(text)
        x[2] += x[1]
        draw()
    except ValueError:
        xtext.set_val("ERROR!")
def submity(text):
    global y
    try:
        y[2] -= y[1]
        y[1] = (float)(text)
        y[2] += y[1]
        draw()
    except ValueError:
        ytext.set_val("ERROR!")
def submitz(text):
    global z
    try:
        z[2] -= z[1]
        z[1] = (float)(text)
        z[2] += z[1]
        draw()
    except ValueError:
        ztext.set_val("ERROR!")

def submitvx(text):
    global x
    try:
        x[2] = (float)(text) + x[1]
        print("x array has been changed.")
    except ValueError:
        vxtext.set_val("ERROR!")
    finally:
        draw()
def submitvy(text):
    global y
    try:
        y[2] = (float)(text) + y[1]
        print("y array has been changed.")
    except ValueError:
        vytext.set_val("ERROR!")
    finally:
        draw()
def submitvz(text):
    global z
    try:
        z[2] = (float)(text) + z[1]
        print("z array has been changed.")
    except ValueError:
        vztext.set_val("ERROR!")
    finally:
        draw()

k = 1.0
dt = 0.1
omega = np.array([0., 0., 0.01])
a0 = np.array([0., 0., 0.])

p, = ax.plot(x, y, z)
p.set_visible(True)

pause = True
FirFlag = True
frms = 0
def FrmData():
    global frms, pause
    if not pause:
        frms += 1
        yield frms
    else:
        time.sleep(0.05)

def f(v, x):
    a = a0 + np.cross(omega, np.cross(omega, x)) + np.cross(2 * omega,  v) 
    return a

def acceleration(t, state):
    r, v = state[:3], state[3:]
    R = np.linalg.norm(r, ord=2)
    # V = np.linalg.norm(v, ord=2)
    # Ep = -GM / R
    # Ek = 0.5 * V ** 2
    # print(Ep + Ek)
    if R == 0 or R == np.nan:raise Exception("Error! invalid R value!")
    a = -GM / (R**3) * r
    return np.concatenate((v, a))

def solve(x0, v0, dt):
    initial_state = np.concatenate((x0, v0))

    t_span = (0, dt)
    dt_min = 0.01
    dt_max = 3600
    max_step = dt_max

    # 使用RK45求解器
    solution = solve_ivp(acceleration, t_span, initial_state, method='RK45', 
                        atol=1e-3, rtol=1e-3, max_step=max_step)
    return solution.y

def update(frms):
    global k, dt, p, x, y, z, v
    print(frms)
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
    if frms >= 100:x, y, z = x[1:], y[1:], z[1:]
    p.set_data_3d(x, y, z)
    return (p,)

   

ani = animation.FuncAnimation(fig2, update, frames=FrmData)

def start(event):
    ax.clear()
    global p, v, x, y, z, pause, FirFlag
    if FirFlag==True:
        v[0] = x[2] - x[1]
        v[1] = y[2] - y[1]
        v[2] = z[2] - z[1]
        x = np.delete(x, 0)
        y = np.delete(y, 0)
        z = np.delete(z, 0)
        x = np.delete(x, 1)
        y = np.delete(y, 1)
        z = np.delete(z, 1)
        FirFlag = False
    xtext.set_active(False)
    ytext.set_active(False)
    ztext.set_active(False)
    vxtext.set_active(False)
    vytext.set_active(False)
    vztext.set_active(False)
    p.set_visible(True)
    pause = False
    p.set_data_3d(x[-1], y[-1], z[-1])
    fig2.canvas.draw_idle()

def stop(event):
    global pause
    pause = True
    xtext.set_active(True)
    ytext.set_active(True)
    ztext.set_active(True)
    vxtext.set_active(True)
    vytext.set_active(True)
    vztext.set_active(True)

xtext.on_submit(submitx)
ytext.on_submit(submity)
ztext.on_submit(submitz)
vxtext.on_submit(submitvx)
vytext.on_submit(submitvy)
vztext.on_submit(submitvz)
Startbtn.on_clicked(start)
Stopbtn.on_clicked(stop)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()