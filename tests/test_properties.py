import random

import array_api_strict as aa
import hypothesis
import hypothesis.strategies as st
import pytest

from rotate import Rotation, skew

aa.set_array_api_strict_flags(api_version="2023.12")


def quaternions_are_equal(q1, q2):
    return (q1[0] == pytest.approx(q2[0]) and q1[1] == pytest.approx(q2[1]) and q1[2] == pytest.approx(q2[2]) and q1[3] == pytest.approx(q2[3])) or (
        q1[0] == pytest.approx(-q2[0]) and q1[1] == pytest.approx(-q2[1]) and q1[2] == pytest.approx(-q2[2]) and q1[3] == pytest.approx(-q2[3])
    )


def test_skew():
    a = aa.reshape(aa.asarray([random.random() for _ in range(300)]), (100, 3))

    b = skew(a)
    assert b.shape == (100, 3, 3)


@pytest.fixture
def quat():
    numbers = aa.reshape(aa.asarray([random.random() for _ in range(1200)]), (300, 4))
    return numbers / aa.linalg.vector_norm(numbers, axis=-1, keepdims=True)


def test_index(quat):
    R = Rotation.from_quat(quat)
    R0 = R[0]

    assert R0.as_quat().shape == (4,)
    q = R0.as_quat()
    assert quaternions_are_equal(q, R.as_quat()[0, :])


def test_concatenate(quat):
    R1 = Rotation(quat[::2, :])
    R2 = Rotation(quat[1::2, :])

    q = Rotation.concatenate([R1, R2]).as_quat()

    q_exp = aa.concat([quat[::2, :], quat[1::2, :]])

    for i in range(q.shape[0]):
        assert quaternions_are_equal(q[i, :], q_exp[i, :])


def test_random():
    q = Rotation.random(aa, 100).as_quat()

    assert q.shape == (100, 4)
    assert aa.linalg.vector_norm(q, axis=-1) == pytest.approx(1.0)


def test_identity():
    q1 = Rotation.identity(aa).as_quat()
    assert q1.shape == (4,)

    q = Rotation.identity(aa, 37).as_quat()

    assert q.shape == (37, 4)
    # assert aa.allclose(q, aa.asarray([[0, 0, 0, 1.0]]))
    assert aa.all(q[:, 0] == 0)
    assert aa.all(q[:, 1] == 0)
    assert aa.all(q[:, 2] == 0)
    assert aa.all(q[:, 3] == 1.0)

    q = Rotation.identity(aa, (13, 36)).as_quat()

    assert q.shape == (13, 36, 4)
    assert aa.all(q[:, :, 0] == 0)
    assert aa.all(q[:, :, 1] == 0)
    assert aa.all(q[:, :, 2] == 0)
    assert aa.all(q[:, :, 3] == 1.0)


@hypothesis.given(st.lists(st.floats(min_value=-1e153, max_value=1e153), min_size=4, max_size=4))
def test_unit_norm_quaternion(values):
    # Returned quaternions have unit norm, no matter the input
    # except 0 0 0 0 and other really small ones
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    assert aa.linalg.vector_norm(rot.as_quat()) == pytest.approx(1.0)


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_double_inverse(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    rot2 = rot.inv().inv()

    # exactly equal since the inversing is simply component sign flipping
    assert aa.all(rot.as_quat() == rot2.as_quat())


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_rotation_matrix_properties(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    # determinant of a rotation matrix is 1
    # thus it preserves lengths
    assert aa.linalg.det(rot.as_matrix()) == pytest.approx(1.0)

    # inverse of a rotation matrix is its transpose
    R = rot.as_matrix()
    Rt = rot.inv().as_matrix()

    assert aa.all(R == aa.linalg.matrix_transpose(Rt))

    # and that is in fact the inverse
    res = R @ Rt
    assert res[0, 0] == pytest.approx(1.0)
    assert res[0, 1] == pytest.approx(0.0)
    assert res[0, 2] == pytest.approx(0.0)
    assert res[1, 0] == pytest.approx(0.0)
    assert res[1, 1] == pytest.approx(1.0)
    assert res[1, 2] == pytest.approx(0.0)
    assert res[2, 0] == pytest.approx(0.0)
    assert res[2, 1] == pytest.approx(0.0)
    assert res[2, 2] == pytest.approx(1.0)


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_to_matrix_and_back(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    mat = rot.as_matrix()
    rot2 = Rotation.from_matrix(mat)

    # We don't compare to inp, which may not be normalized
    q1 = rot.as_quat()
    q2 = rot2.as_quat()

    assert quaternions_are_equal(q1, q2)


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_to_rotvec_and_back(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    rotvec = rot.as_rotvec()
    rot2 = Rotation.from_rotvec(rotvec)

    q1 = rot.as_quat()
    q2 = rot2.as_quat()

    assert quaternions_are_equal(q1, q2)


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_to_mrp_and_back(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    mrp = rot.as_mrp()
    rot2 = Rotation.from_mrp(mrp)

    q1 = rot.as_quat()
    q2 = rot2.as_quat()

    assert quaternions_are_equal(q1, q2)


@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_to_eulerXYZ_and_back(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    eul = rot.as_euler("XYZ")
    rot2 = Rotation.from_euler("XYZ", eul)

    q1 = rot.as_quat()
    q2 = rot2.as_quat()

    assert quaternions_are_equal(q1, q2)


# Weirdly, this fails with small strange differences - but so does scipy
@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_to_eulerZXZ_and_back(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    eul = rot.as_euler("ZXZ")
    rot2 = Rotation.from_euler("ZXZ", eul)

    q1 = rot.as_quat()
    q2 = rot2.as_quat()

    assert quaternions_are_equal(q1, q2)


# random quat and vector
@hypothesis.given(
    st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4), st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3)
)
def test_rotate_vector(qvalues, vvalues):
    qinp = aa.asarray(qvalues)
    vinp = aa.asarray(vvalues)
    hypothesis.assume(aa.linalg.vector_norm(qinp) > 1e-3)  # not too unreasonable small

    rot = Rotation.from_quat(qinp)

    v = rot.apply(vinp)
    assert v.shape == (3,)
    # Length is preserved
    assert aa.linalg.vector_norm(v) == pytest.approx(aa.linalg.vector_norm(vinp))

    # Rotating back should give the same vector
    v2 = rot.inv().apply(v)
    assert v2[0] == pytest.approx(vinp[0])
    assert v2[1] == pytest.approx(vinp[1])
    assert v2[2] == pytest.approx(vinp[2])


# rotvec with 2*pi added should be the same
# TODO it isn't tho, there are small differences - fix or?
# @hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3))
# def test_rotvec_periodicity(values):
#     inp = aa.asarray(values)

#     # Doesn't work for 0
#     hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

#     rot = Rotation.from_rotvec(inp)

#     # Add 2*pi to the rotvec on the same axis
#     inp2 = inp + 2 * aa.pi * inp / aa.linalg.vector_norm(inp)
#     rot2 = Rotation.from_rotvec(inp2)

#     q1 = rot.as_quat()
#     q2 = rot2.as_quat()

#     assert quaternions_are_equal(q1, q2)


# Quaternions that have opposite sign produce same rotation matrix
@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_opposite_quaternions(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    # Invert the quaternion
    inp2 = -inp
    rot2 = Rotation.from_quat(inp2)

    # The rotation matrices should be the same
    assert aa.all(rot.as_matrix() == rot2.as_matrix())


# Quaternions multiply by inverse produce identity
@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_quaternion_inverse(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    # Multiply by inverse
    rot2 = rot * rot.inv()

    q = rot2.as_quat()
    assert q[0] == pytest.approx(0.0)
    assert q[1] == pytest.approx(0.0)
    assert q[2] == pytest.approx(0.0)
    assert q[3] == pytest.approx(1.0)


# zeroth power is identity
@hypothesis.given(st.lists(st.floats(min_value=-1, max_value=1), min_size=4, max_size=4))
def test_quaternion_power_identity(values):
    inp = aa.asarray(values)
    hypothesis.assume(aa.linalg.vector_norm(inp) != 0)

    rot = Rotation.from_quat(inp)

    rot2 = rot**0

    q = rot2.as_quat()
    assert q[0] == pytest.approx(0.0)
    assert q[1] == pytest.approx(0.0)
    assert q[2] == pytest.approx(0.0)
    assert q[3] == pytest.approx(1.0)
