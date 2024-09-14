import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRotation

from rotate import Rotation, skew


@pytest.fixture
def srotations():
    return sRotation.random(100)


@pytest.fixture
def quat(srotations):
    return srotations.as_quat()


@pytest.fixture
def matrix(srotations):
    return srotations.as_matrix()


@pytest.fixture
def mrp(srotations):
    return srotations.as_mrp()


@pytest.fixture
def rotvec(srotations):
    return srotations.as_rotvec()


@pytest.fixture
def euler_xyz(srotations):
    return srotations.as_euler("xyz")


@pytest.fixture
def euler_xyz_deg(srotations):
    return srotations.as_euler("xyz", degrees=True)


@pytest.fixture
def euler_zxz(srotations):
    return srotations.as_euler("zxz")


@pytest.fixture
def euler_XYZ(srotations):
    return srotations.as_euler("XYZ")


def test_to_rot_matrix(quat, matrix):
    M = Rotation(quat).as_matrix()
    assert M.shape == (quat.shape[0], 3, 3)

    assert np.allclose(M, matrix)


def test_to_euler(quat, euler_xyz, euler_XYZ, euler_zxz):
    e = Rotation(quat).as_euler("xyz")
    assert e.shape == euler_xyz.shape

    assert np.allclose(e, euler_xyz)

    e = Rotation(quat).as_euler("XYZ")
    assert e.shape == euler_XYZ.shape

    assert np.allclose(e, euler_XYZ)

    e = Rotation(quat).as_euler("zxz")
    assert e.shape == euler_zxz.shape

    assert np.allclose(e, euler_zxz)


def quaternions_are_equal(q1, q2):
    return np.allclose(q1, q2) or np.allclose(q1, -q2)


def test_matrix_to_quat(matrix, quat):
    q = Rotation.from_matrix(matrix).as_quat()

    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])


def test_euler_to_quat(euler_xyz, euler_XYZ, euler_zxz, quat):
    q = Rotation.from_euler("xyz", euler_xyz).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])

    q = Rotation.from_euler("XYZ", euler_XYZ).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])

    q = Rotation.from_euler("zxz", euler_zxz).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])


def test_euler_deg_to_quat(euler_xyz_deg, quat):
    q = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])


def test_quat_to_mrp(quat, mrp):
    m = Rotation(quat).as_mrp()
    assert m.shape == mrp.shape

    assert np.allclose(m, mrp)


def test_mrp_to_quat(mrp, quat):
    q = Rotation.from_mrp(mrp).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])


def test_rotvec_to_quat(rotvec, quat):
    q = Rotation.from_rotvec(rotvec).as_quat()
    assert q.shape == quat.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], quat[i])


def test_quat_to_rotvec(quat, rotvec):
    r = Rotation(quat).as_rotvec()
    assert r.shape == rotvec.shape

    assert np.allclose(r, rotvec)


def test_mul(quat):
    R1 = Rotation(quat[::2])
    R2 = Rotation(quat[1::2])

    q = (R1 * R2).as_quat()

    q_exp = (sRotation.from_quat(quat[::2]) * sRotation.from_quat(quat[1::2])).as_quat()

    assert q.shape == q_exp.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], q_exp[i])


def test_inv(quat):
    q = Rotation(quat).inv().as_quat()

    q_exp = sRotation.from_quat(quat).inv().as_quat()

    assert q.shape == q_exp.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], q_exp[i])


@pytest.mark.parametrize("n", range(-1, 4))
def test_pow(quat, n):
    q = (Rotation(quat) ** n).as_quat()

    q_exp = (sRotation.from_quat(quat) ** n).as_quat()

    assert q.shape == q_exp.shape

    for i in range(len(q)):
        assert quaternions_are_equal(q[i], q_exp[i])


def test_magnitude(quat):
    m = Rotation(quat).magnitude()

    m_exp = sRotation.from_quat(quat).magnitude()

    assert m.shape == m_exp.shape

    assert np.allclose(m, m_exp)


def test_mean(quat):
    qmean = Rotation(quat).mean().as_quat()

    qmean_exp = sRotation.from_quat(quat).mean().as_quat()

    assert qmean.shape == qmean_exp.shape

    assert quaternions_are_equal(qmean, qmean_exp)


def test_apply_single_vec(quat):
    # Test with single vector
    R = Rotation(quat)
    v = np.array([1, 0, 0])
    v_rot = R.apply(v)
    v_rot_exp = sRotation.from_quat(quat).apply(v)

    assert np.allclose(v_rot, v_rot_exp)


def test_apply_many_vec(quat):
    # Test with as many vectors as quaternions
    R = Rotation.from_quat(quat)
    v = np.random.normal(size=(quat.shape[0], 3))
    v_rot = R.apply(v)
    v_rot_exp = sRotation.from_quat(quat).apply(v)

    assert np.allclose(v_rot, v_rot_exp)


def test_apply_single_rot(quat):
    # Test with single rotation and many vectors
    v = np.random.normal(size=(100, 3))
    R = Rotation(quat[0])
    v_rot = R.apply(v)
    v_rot_exp = sRotation.from_quat(quat[0]).apply(v)

    assert np.allclose(v_rot, v_rot_exp)
