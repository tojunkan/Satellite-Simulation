# print("Rotation module loaded")

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def Rotating(vector: np.ndarray, n: np.ndarray, theta):
    # 构建 K 矩阵
    K = np.array([[0, -n[2], n[1]],
                [n[2], 0, -n[0]],
                [-n[1], n[0], 0]])
    # 构建 K^2 矩阵
    K2 = np.dot(K, K)
    # 计算旋转矩阵 R
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K2
    # 初始向量
    v = vector
    # 旋转后的向量
    v_rotated = np.dot(R, v)
    return v_rotated

def CalcQuiver3D(x, y, z, u, v, w, arrow_length_ratio, length=1):

    def calc_arrows(UVW):
        # get unit direction vector perpendicular to (u, v, w)
        x = UVW[:, 0]
        y = UVW[:, 1]
        norm = np.linalg.norm(UVW[:, :2], axis=1)
        x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
        y_p = np.divide(-x,  norm, where=norm != 0, out=np.ones_like(x))
        # compute the two arrowhead direction unit vectors
        rangle = np.radians(15)
        c = np.cos(rangle)
        s = np.sin(rangle)
        # construct the rotation matrices of shape (3, 3, n)
        r13 = y_p * s
        r32 = x_p * s
        r12 = x_p * y_p * (1 - c)
        Rpos = np.array(
            [[c + (x_p ** 2) * (1 - c), r12, r13],
                [r12, c + (y_p ** 2) * (1 - c), -r32],
                [-r13, r32, np.full_like(x_p, c)]])
        # opposite rotation negates all the sin terms
        Rneg = Rpos.copy()
        Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
        # Batch n (3, 3) x (3) matrix multiplications ((3, 3, n) x (n, 3)).
        Rpos_vecs = np.einsum("ij...,...j->...i", Rpos, UVW)
        Rneg_vecs = np.einsum("ij...,...j->...i", Rneg, UVW)
        # Stack into (n, 2, 3) result.
        return np.stack([Rpos_vecs, Rneg_vecs], axis=1)

    shaft_dt = np.array([0., length], dtype=float)
    arrow_dt = shaft_dt * arrow_length_ratio

    # 计算箭头的线段
    XYZ = np.column_stack([x, y, z])
    UVW = np.column_stack([u, v, w]).astype(float)
    # 设置轴标签
    # compute the shaft lines all at once with an outer product
    shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
    # compute head direction vectors, n heads x 2 sides x 3 dimensions
    head_dirs = calc_arrows(UVW)
    # compute all head lines at once, starting from the shaft ends
    heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
    # stack left and right head lines together
    heads = heads.reshape((len(arrow_dt), -1, 3))
    # transpose to get a list of lines
    heads = heads.swapaxes(0, 1)

    lines = [*shafts, *heads[::2], *heads[1::2]]

    return lines


# # 创建3D绘图对象
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制坐标轴
# ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
# ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
# ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

# # 绘制旋转轴
# ax.quiver(0, 0, 0, *n, color='m', arrow_length_ratio=0.1)

# # 使用 quiver 绘制初始向量和旋转后的向量
# ax.quiver(0, 0, 0, *v, color='c', arrow_length_ratio=0.1)
# ax.quiver(0, 0, 0, *v_rotated, color='y', arrow_length_ratio=0.1)

# def set_axes_equal(ax: plt.Axes):
#     x = y = z = np.array([-1, 1])
#     xmean = np.mean(x)
#     ymean = np.mean(y)
#     zmean = np.mean(z)

#     xrange = np.max(x) - np.min(x)
#     yrange = np.max(y) - np.min(y)
#     zrange = np.max(z) - np.min(z)
#     lim = max(xrange, yrange, zrange) / 2.0


#     if lim != 0 and xmean != None and ymean != None and zmean != None and lim != None:
#         ax.set_xlim(xmean - lim, xmean + lim)
#         ax.set_ylim(ymean - lim, ymean + lim)
#         ax.set_zlim(zmean - lim, zmean + lim)
#     else: 
#         ax.set_xlim(-6.371e6, 6.371e6)
#         ax.set_ylim(-6.371e6, 6.371e6)
#         ax.set_zlim(-6.371e6, 6.371e6) 

#     limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d(),])
#     origin = np.mean(limits, axis=1)
#     radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
#     x0, y0, z0 = origin
#     ax.set_xlim3d([x0 - radius, x0 + radius])
#     ax.set_ylim3d([y0 - radius, y0 + radius])
#     ax.set_zlim3d([z0 - radius, z0 + radius])
# ax.set_box_aspect((1, 1, 1))

# # 设置坐标轴范围和标签
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # 设置视角
# ax.view_init(elev=90-54.7356103101115, azim=45)

# plt.show()

# print("End of Rotation module")