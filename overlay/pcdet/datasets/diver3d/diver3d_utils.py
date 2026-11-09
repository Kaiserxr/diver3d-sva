"""
Box conversions between Diver3D (full SO(3)) and OpenPCDet (yaw-only). numpy only.

OpenPCDet box: (x, y, z, dx, dy, dz, heading), z = box centre, dz vertical.
Diver3D box:   centre, dimensions (length, width, height) along R = [ex ey ez], quaternion (w, x, y, z).
"""

import numpy as np

_CORNERS = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])


def quat_to_rotmat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def so3_to_yaw_box(center, dimensions, quaternion):
    """Tightest yaw-only box enclosing the SO(3) box.

    Heading is the principal horizontal direction of the 8 corners, so dx is the
    longest horizontal extent (a lying diver gets dx ~ body length, dz ~ thickness).
    """
    R = quat_to_rotmat(quaternion)
    corners = (_CORNERS * np.asarray(dimensions)) @ R.T + np.asarray(center)
    xy = corners[:, :2] - corners[:, :2].mean(0)
    _, vecs = np.linalg.eigh(xy.T @ xy)
    d = vecs[:, 1]
    yaw = np.arctan2(d[1], d[0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = corners @ Rz                      # corners in the yawed frame
    lo, hi = local.min(0), local.max(0)
    ctr = Rz @ ((lo + hi) / 2)
    return np.array([*ctr, *(hi - lo), yaw], dtype=np.float32)


def yaw_box_to_record(box, score):
    """OpenPCDet yaw box -> Diver3D submission record (R = Rz(yaw), so ez is sonar z-up)."""
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    return {
        "center": [x, y, z],
        "dimensions": [dx, dy, dz],
        "quaternion": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        "score": float(score),
    }
