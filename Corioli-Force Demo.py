import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

omega = 10
xi = np.array([0])
yi = np.array([-1])
vi = np.array([1, 0])

xe = np.array([0])
ye = np.array([-1])

xr = np.array([0])
yr = np.array([0])

xr2 = np.array([0])
yr2 = np.array([-1])
n = 200
t = np.linspace(0, 2, n)
a = np.array([1, 0])

pi, = ax.plot(xi, yi)
pe, = ax.plot(xe, ye)
pr, = ax.plot(xr, yr)
pr2, = ax.plot(xr2, yr2)


ax.set_aspect('equal', adjustable='box')

ax.grid(True)

def setlim():
    tx = np.concatenate((xi, xe, xr, xr2))
    ty = np.concatenate((yi, ye, yr, yr2))
    xrange = np.max(tx) + 2 - np.min(tx)
    yrange = np.max(ty) + 2 - np.min(ty)
    ramge = max(xrange, yrange) / 2

    xmean = np.mean(tx)
    ymean = np.mean(ty)

    ax.set_xlim(xmean-ramge, xmean+ramge)
    ax.set_ylim(ymean-ramge, ymean+ramge)
    ax.set_aspect('equal', adjustable='box')

def update(i):
    global xi, yi, vi, xe, ye, xr, yr, xr2, yr2
    dxi = xi[0] + 0.5 * a[0] * t[i] * t[i] + vi[0] * t[i]
    dyi = yi[0] + 0.5 * a[1] * t[i] * t[i] + vi[1] * t[i]
    xi = np.append(xi, dxi)
    yi = np.append(yi, dyi)

    xe = np.append(xe, np.cos(omega * t[i]- 0.5 * np.pi))
    ye = np.append(ye, np.sin(omega * t[i]- 0.5 * np.pi))

    xr = np.append(xr, xi[-1] - xe[-1])
    yr = np.append(yr, yi[-1] - ye[-1])

    xr2 = np.append(xr2, xi[-1] * np.cos(omega * t[i]) - yi[-1] * np.sin(omega * t[i]))
    yr2 = np.append(yr2, xi[-1] * np.sin(omega * t[i]) + yi[-1] * np.cos(omega * t[i]))

    print(np.sqrt(xi[-1] ** 2 + yi[-1] ** 2), np.sqrt(xr2[-1] ** 2 + yr2[-1] ** 2))

    pi.set_data(xi, yi)
    pe.set_data(xe, ye)
    pr.set_data(xr, yr)
    pr2.set_data(xr2, yr2)

    setlim()
    return pi, pe, pr, pr2,

ani = FuncAnimation(fig=fig, func=update, frames=n, repeat=False)

ani.save(r'C:\Users\UniDu\Desktop\Corioli-Force Demo.gif', writer='pillow', fps=30)
plt.show()