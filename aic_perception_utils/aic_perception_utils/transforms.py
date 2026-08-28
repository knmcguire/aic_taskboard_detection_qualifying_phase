"""Shared rotation and quaternion helpers (xyzw convention)."""

import numpy as np


def quaternion_to_rotation_matrix(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (m21 - m12) * scale
        y = (m02 - m20) * scale
        z = (m10 - m01) * scale
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / max(np.linalg.norm(quaternion), 1e-12)
