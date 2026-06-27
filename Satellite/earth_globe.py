"""
EarthGlobe — 地球3D可视化节点

封装球体网格 + 线框 + 纹理/自转后门。

后门说明：
- UV 纹理坐标已预置在网格数据中，后续只需 set_texture(path) 即可贴图
- 自转 MatrixTransform 已挂载，后续每帧 rotate(angle) 即可自转
"""

import numpy as np
from vispy import scene
from vispy.visuals import MeshVisual
from vispy.visuals.transforms import MatrixTransform


def _build_uv_sphere(radius, rows, cols):
    """构建带 UV 纹理坐标的球体网格。

    Parameters
    ----------
    radius : float
        球体半径（米）
    rows : int
        纬度分段数
    cols : int
        经度分段数

    Returns
    -------
    dict
        vertices: (N, 3) float32  顶点坐标
        faces:    (M, 3) uint32   三角形面索引
        texcoords: (N, 2) float32 UV 纹理坐标 ★ 贴图后门
    """
    # 纬度 0→π（北极→南极），经度 0→2π
    phi = np.linspace(0, np.pi, rows + 1)
    theta = np.linspace(0, 2 * np.pi, cols + 1)

    vertices = []
    texcoords = []
    for i in range(rows + 1):
        sin_phi = np.sin(phi[i])
        cos_phi = np.cos(phi[i])
        for j in range(cols + 1):
            x = radius * sin_phi * np.cos(theta[j])
            y = radius * sin_phi * np.sin(theta[j])
            z = radius * cos_phi
            vertices.append([x, y, z])
            # UV: u=经度, v=纬度（0-1 归一化）
            texcoords.append([j / cols, i / rows])

    vertices = np.array(vertices, dtype=np.float32)
    texcoords = np.array(texcoords, dtype=np.float32)

    # 每个网格单元两个三角形
    faces = []
    for i in range(rows):
        for j in range(cols):
            a = i * (cols + 1) + j
            b = a + 1
            c = (i + 1) * (cols + 1) + j
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])

    faces = np.array(faces, dtype=np.uint32)
    return {'vertices': vertices, 'faces': faces, 'texcoords': texcoords}


def _extract_edges(faces):
    """从三角面提取唯一边（用于线框渲染）。"""
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return np.array(sorted(edges), dtype=np.uint32)


class EarthGlobe(scene.Node):
    """地球 3D 可视化节点。

    场景图结构：
        EarthGlobe (self)
          └── _rotation_node (MatrixTransform) ★ 自转后门
                ├── _surface (MeshVisual)       — 半透明蓝色表面
                └── _wireframe (LineVisual)     — 经纬网格线
    """

    def __init__(self, radius=6.371e6, rows=60, cols=120,
                 face_color=(0.15, 0.35, 0.65, 0.25),
                 wire_color=(0.5, 0.7, 1.0, 0.6)):
        super().__init__()
        self.radius = radius
        self._rows = rows
        self._cols = cols

        # ---- 自转变换节点 ★ 后门 ----
        self._rotation_node = scene.Node(name='earth_rotation')
        self._rotation_node.parent = self
        self._rotation_node.transform = MatrixTransform()

        # ---- 构建网格数据 ----
        mesh_data = _build_uv_sphere(radius, rows, cols)
        self._vertices = mesh_data['vertices']
        self._faces = mesh_data['faces']
        self._texcoords = mesh_data['texcoords']  # ★ 纹理后门

        # ---- 半透明表面 ----
        vertex_colors = np.tile(
            np.array(face_color, dtype=np.float32),
            (len(self._vertices), 1)
        )
        self._surface = scene.visuals.Mesh(
            vertices=self._vertices,
            faces=self._faces,
            vertex_colors=vertex_colors,
            shading='smooth',
        )
        self._surface.parent = self._rotation_node
        self._surface.set_gl_state(
            depth_test=True,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
            cull_face=True,
        )

        # ---- 线框（经纬网格） ----
        edges = _extract_edges(self._faces)
        edge_vertices = self._vertices[edges]  # (E, 2, 3)
        wire_pos = edge_vertices.reshape(-1, 3)

        # 构建 segment 连接索引
        wire_connect = np.arange(len(wire_pos), dtype=np.uint32).reshape(-1, 2)

        self._wireframe = scene.visuals.Line(
            pos=wire_pos,
            connect=wire_connect,
            color=wire_color,
            width=1.0,
            method='gl',
        )
        self._wireframe.parent = self._rotation_node

        # 保存材质参数供后续使用
        self._face_color = face_color
        self._wire_color = wire_color

    # ========== ★ 后门方法 ==========

    def set_texture(self, image_path):
        """加载地球纹理贴图 ★ 后门

        当前为空实现。后续加载一张地球纹理图（如 NASA Blue Marble），
        应用到球体表面即可显示大陆和海洋。

        Usage (future):
            earth.set_texture('earth_blue_marble.jpg')
        """
        # TODO: 阶段2实现
        # from vispy.io import load_data_file, read_png
        # texture_image = read_png(image_path)
        # self._surface.set_data(
        #     vertices=self._vertices,
        #     faces=self._faces,
        #     texcoords=self._texcoords,
        # )
        # self._surface.texture = texture_image
        raise NotImplementedError("纹理贴图将在阶段2实现")

    def rotate(self, angle, axis=(0, 0, 1)):
        """绕指定轴旋转地球 ★ 后门（自转用）

        Parameters
        ----------
        angle : float
            旋转角度（弧度）
        axis : tuple
            旋转轴（默认 Z 轴 = 地球自转轴）
        """
        self._rotation_node.transform.rotate(angle, axis)

    def set_opacity(self, alpha):
        """调整表面透明度。"""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self._face_color = (*self._face_color[:3], alpha)
        vertex_colors = np.tile(
            np.array(self._face_color, dtype=np.float32),
            (len(self._vertices), 1)
        )
        self._surface.set_data(
            vertices=self._vertices,
            faces=self._faces,
            vertex_colors=vertex_colors,
        )
        # set_data 后必须重新应用 GL 混合状态，否则透明度不生效
        self._surface.set_gl_state(
            depth_test=True,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'),
            cull_face=True,
        )
        self._surface.update()

    def set_wireframe_visible(self, visible):
        """显示/隐藏经纬线框。"""
        self._wireframe.visible = visible

    def load_continents(self, continents):
        """加载大陆三角网格（每个为 (name, vertices, triangles, color)）。

        大陆网格挂载在自转节点下，随地球自转一起旋转。
        """
        self._continent_meshes = []
        for name, verts, tris, color in continents:
            face_colors = np.tile(
                np.array([*color, 1.0], dtype=np.float32), (len(tris), 1)
            )
            mesh = scene.visuals.Mesh(
                vertices=verts.astype(np.float32),
                faces=tris.astype(np.uint32),
                face_colors=face_colors,
                shading='flat',
            )
            mesh.parent = self._rotation_node
            mesh.set_gl_state(depth_test=True, cull_face=False)
            mesh.visible = True
            self._continent_meshes.append(mesh)

    def set_continents_visible(self, visible):
        """显示/隐藏大陆网格。"""
        for m in getattr(self, '_continent_meshes', []):
            m.visible = visible


# ========== 自测 ==========
if __name__ == '__main__':
    from vispy import app
    app.use_app('pyqt5')

    canvas = scene.SceneCanvas(
        keys='interactive',
        size=(800, 600),
        title='EarthGlobe 自测',
        show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(
        fov=45, elevation=30, azimuth=60,
        distance=3e7,  # 约 5 倍地球半径
    )

    earth = EarthGlobe(radius=6.371e6, rows=80, cols=160)
    earth.parent = view.scene

    # 测试自转
    def on_timer(ev):
        earth.rotate(0.01, (0, 0, 1))

    timer = app.Timer(interval=0.03, connect=on_timer, start=True)

    # 坐标轴辅助线
    axis_len = 1.5e7
    for axis, color in zip([(axis_len, 0, 0), (0, axis_len, 0), (0, 0, axis_len)],
                           ['red', 'green', 'blue']):
        line = scene.visuals.Line(
            pos=np.array([[0, 0, 0], axis]),
            color=color,
            width=2,
        )
        line.parent = view.scene

    canvas.app.run()
