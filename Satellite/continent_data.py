"""
continent_data.py — 大陆 3D 网格数据

数据来源：DrawCoastlinesDemo.py 手工三角化成果
预处理流程：海岸线 → 转到南极 → 平面 Delaunay → 逆旋转 → 球面三角网格

每个大陆包含 (N,3) 顶点（地心笛卡尔, m）和 (M,3) 三角形索引。
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    return np.load(os.path.join(_DIR, f'continent_{name}.npy'))


# 颜色：不同大陆用不同绿色调
CONTINENTS = [
    ('Africa',     _load('afri'),      _load('afri_tria'),      (0.20, 0.48, 0.12)),
    ('Eurasia',    _load('eurasia'),   _load('euras_tria'),     (0.22, 0.52, 0.14)),
    ('Australia',  _load('austra'),    _load('austra_tria'),    (0.45, 0.35, 0.12)),
    ('N.America',  _load('nor_amer'),  _load('nor_amer_tria'),  (0.18, 0.45, 0.12)),
    ('S.America',  _load('sou_amer'),  _load('sou_amer_tria'),  (0.15, 0.42, 0.10)),
    ('Antarctica', _load('antarct_G'), _load('antarct_G_tria'), (0.85, 0.90, 0.80)),
]
