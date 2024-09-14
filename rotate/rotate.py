import re
from random import Random
from typing import List, Optional, Tuple, Union

import array_api_compat
from typing_extensions import Self


def skew(a):
    # a.shape = (..., 3)
    xp = array_api_compat.array_namespace(a)
    outshape = a.shape[:-1] + (3, 3)
    z = xp.zeros_like(a[..., 0])
    return xp.reshape(xp.stack([z, -a[..., 2], a[..., 1], a[..., 2], z, -a[..., 0], -a[..., 1], a[..., 0], z], axis=-1), outshape)


class Rotation:
    """
    Main class, supposed to work with any array type that conforms to the array API

    """

    _axes = ["x", "y", "z"]

    def __init__(self, quat, normalize=False):
        xp = array_api_compat.array_namespace(quat)
        if normalize:
            quat = quat / xp.linalg.vector_norm(quat, axis=-1, keepdims=True)
        self._quat = quat

    @staticmethod
    def _quat_conj(q):
        xp = array_api_compat.array_namespace(q)
        return xp.concat([-q[..., :3], q[..., 3, None]], axis=-1)

    @staticmethod
    def _quat_mul(q1, q2):
        xp = array_api_compat.array_namespace(q1)
        q13 = q1[..., 3, None] * q2[..., :3] + q2[..., 3, None] * q1[..., :3] + xp.linalg.cross(q1[..., :3], q2[..., :3])
        q4 = q1[..., 3] * q2[..., 3] - xp.vecdot(q1[..., :3], q2[..., :3], axis=-1)
        q = xp.concat([q13, q4[..., None]], axis=-1)
        return q

    @staticmethod
    def _quat_imag_part(q):
        return q[..., :3]

    @staticmethod
    def _quat_scalar_part(q):
        return q[..., 3]

    def __len__(self) -> int:
        xp = array_api_compat.array_namespace(self._quat)
        if self._quat.ndim == 1:
            return 1
        return xp.prod(xp.asarray(self._quat.shape[:-1]))

    def __getitem__(self, key: Union[int, slice]):
        return self.__class__(self._quat[key, :])

    @classmethod
    def from_quat(cls, quat):
        return cls(quat, normalize=True)

    @classmethod
    def from_matrix(cls, M):
        xp = array_api_compat.array_namespace(M)

        trM = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]

        opt1 = xp.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1], 1 + trM], axis=-1)
        opt2 = xp.stack([1 + 2 * M[..., 0, 0] - trM, M[..., 0, 1] + M[..., 1, 0], M[..., 0, 2] + M[..., 2, 0], M[..., 2, 1] - M[..., 1, 2]], axis=-1)
        opt3 = xp.stack([M[..., 0, 1] + M[..., 1, 0], 1 + 2 * M[..., 1, 1] - trM, M[..., 1, 2] + M[..., 2, 1], M[..., 0, 2] - M[..., 2, 0]], axis=-1)
        opt4 = xp.stack([M[..., 0, 2] + M[..., 2, 0], M[..., 1, 2] + M[..., 2, 1], 1 + 2 * M[..., 2, 2] - trM, M[..., 1, 0] - M[..., 0, 1]], axis=-1)

        val1 = trM
        val2 = M[..., 0, 0]
        val3 = M[..., 1, 1]
        val4 = M[..., 2, 2]

        c1 = (val1 >= val2) & (val1 >= val3) & (val1 >= val4)
        c2 = (val2 >= val1) & (val2 >= val3) & (val2 >= val4)
        c3 = (val3 >= val1) & (val3 >= val2) & (val3 >= val4)
        # c4 = (trM > M[..., 0, 0]) & (trM > M[..., 1, 1]) & (trM > M[..., 2, 2])

        q = xp.where(c1[..., None], opt1, xp.where(c2[..., None], opt2, xp.where(c3[..., None], opt3, opt4)))
        # normalize
        return cls(q, normalize=True)

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        xp = array_api_compat.array_namespace(rotvec)
        if degrees:
            rotvec = rotvec * xp.pi / 180.0

        angle = xp.linalg.vector_norm(rotvec, axis=-1, keepdims=True)

        scale1 = xp.sin(angle / 2) / angle
        # Handle small angles
        small = xp.abs(angle) <= 1e-3
        # Taylor expansion
        angle2 = angle * angle
        scale2 = 0.5 - angle2 / 48 + angle2 * angle2 / 3840

        scale = xp.where(small, scale2, scale1)

        c = xp.cos(angle / 2)
        q = xp.concat([scale * rotvec, c], axis=-1)
        # already normalized
        return cls(q)

    @classmethod
    def from_mrp(cls, mrp):
        xp = array_api_compat.array_namespace(mrp)
        q = xp.concat([2 * mrp, 1 - xp.square(xp.linalg.vector_norm(mrp, axis=-1, keepdims=True))], axis=-1)
        return cls(q, normalize=True)

    @staticmethod
    def _check_seq(seq):
        num_axes = len(seq)
        if num_axes < 1 or num_axes > 3:
            raise ValueError("Expected axis specification to be a non-empty " "string of upto 3 characters, got {}".format(seq))

        intrinsic = re.match(r"^[XYZ]{1,3}$", seq) is not None
        extrinsic = re.match(r"^[xyz]{1,3}$", seq) is not None
        if not (intrinsic or extrinsic):
            raise ValueError("Expected axes from `seq` to be from ['x', 'y', " "'z'] or ['X', 'Y', 'Z'], got {}".format(seq))

        if any(seq[i] == seq[i + 1] for i in range(num_axes - 1)):
            raise ValueError("Expected consecutive axes to be different, " "got {}".format(seq))

        return seq.lower(), intrinsic, num_axes

    @classmethod
    def from_euler(cls, seq, euler, degrees=False):
        seq, intrinsic, num_axes = cls._check_seq(seq)

        xp = array_api_compat.array_namespace(euler)

        if degrees:
            euler = euler * xp.pi / 180.0

        result = cls._elementary_quat(seq[0], euler[..., 0])
        for i in range(1, num_axes):
            if intrinsic:
                result = cls._quat_mul(result, cls._elementary_quat(seq[i], euler[..., i]))
            else:
                result = cls._quat_mul(cls._elementary_quat(seq[i], euler[..., i]), result)

        return cls(result, normalize=True)

    @staticmethod
    def _elementary_quat(axis, angles):
        """
        Quaternion for a rotation of `angles` radians about `axis`
        """
        xp = array_api_compat.array_namespace(angles)

        # x-> 0, y-> 1, z-> 2

        ind = Rotation._axes.index(axis)

        # quat = xp.zeros(angles.shape + (4,), dtype=angles.dtype, device=angles.device)
        # quat[..., ind] = xp.sin(angles / 2)
        # quat[..., 3] = xp.cos(angles / 2)

        # Above assignments don't work with jax, so we do this - it is ugly though
        if ind == 0:
            quat = xp.stack([xp.sin(angles / 2), xp.zeros_like(angles), xp.zeros_like(angles), xp.cos(angles / 2)], axis=-1)
        elif ind == 1:
            quat = xp.stack([xp.zeros_like(angles), xp.sin(angles / 2), xp.zeros_like(angles), xp.cos(angles / 2)], axis=-1)
        else:
            quat = xp.stack([xp.zeros_like(angles), xp.zeros_like(angles), xp.sin(angles / 2), xp.cos(angles / 2)], axis=-1)

        return quat

    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        raise NotImplementedError("Not implemented yet.")

    def as_quat(self):
        return self._quat

    def as_matrix(self):
        xp = array_api_compat.array_namespace(self._quat)

        # This assumes that the quaternions are normalized
        q13 = self._quat[..., :3]
        q4 = self._quat[..., 3]

        # dtype and device arguments may be needed for e.g. GPU arrays
        I3 = xp.eye(3)  #  , dtype=self._quat.dtype, device=self._quat.device)
        return (
            (xp.square(q4) - xp.square(xp.linalg.vector_norm(q13, axis=-1)))[..., None, None] * I3
            + 2 * q4[..., None, None] * skew(q13)
            + 2 * q13[..., :, None] * q13[..., None, :]
        )

    def as_rotvec(self, degrees=False):
        xp = array_api_compat.array_namespace(self._quat)
        # We pick the sign of the scalar part to be positive
        # Ensures that the angle is between 0 and pi

        # xp.sign returns 0 for 0, so we need to calculate it
        ones = xp.ones_like(self._quat[..., 3])
        sign = xp.where(self._quat[..., 3] > 0, 1 * ones, -1 * ones)

        norm123 = xp.linalg.vector_norm(self._quat[..., :3], axis=-1)
        angle = 2 * xp.atan2(norm123, sign * self._quat[..., 3])

        scale1 = angle / xp.sin(angle / 2)
        # Handle small angles
        small = xp.abs(angle) <= 1e-3
        # Taylor expansion
        angle2 = angle * angle
        scale2 = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
        scale = xp.where(small, scale2, scale1)

        rotvec = scale[..., None] * sign[..., None] * self._quat[..., :3]
        if degrees:
            rotvec = rotvec * 180 / xp.pi
        return rotvec

    def as_mrp(self):
        xp = array_api_compat.array_namespace(self._quat)
        # We pick the sign of the scalar part to be positive
        # This makes the MRP correspond to the smaller rotation of the two possible
        ones = xp.ones_like(self._quat[..., 3])
        sign = xp.where(self._quat[..., 3] > 0, 1 * ones, -1 * ones)

        return sign[..., None] * self._quat[..., :3] / (1 + sign * self._quat[..., 3])[..., None]

    def as_euler(self, seq, degrees=False):
        # Based on scipy's implementation
        # But vectorized
        xp = array_api_compat.array_namespace(self._quat)
        seq, intrinsic, _ = self._check_seq(seq)

        if intrinsic:
            seq = seq[::-1]

        i = Rotation._axes.index(seq[0])
        j = Rotation._axes.index(seq[1])
        k = Rotation._axes.index(seq[2])

        symmetric = i == k
        if symmetric:
            k = 3 - i - j  # get third axis

        # Check if permutation is even or odd
        sign = (i - j) * (j - k) * (k - i) // 2

        temp = xp.stack([self._quat[..., 3], self._quat[..., i], self._quat[..., j], self._quat[..., k] * sign], axis=-1)
        if not symmetric:
            temp += xp.stack([-self._quat[..., j], self._quat[..., k] * sign, self._quat[..., 3], -self._quat[..., i]], axis=-1)

        # Step 2
        # Compute second angle
        angles1 = 2 * xp.atan2(xp.hypot(temp[..., 2], temp[..., 3]), xp.hypot(temp[..., 0], temp[..., 1]))

        # Step 3
        # compute first and third angles, according to case
        half_sum = xp.atan2(temp[..., 1], temp[..., 0])
        half_diff = xp.atan2(temp[..., 3], temp[..., 2])

        c1 = xp.abs(angles1) < 1e-7
        c2 = xp.abs(angles1 - xp.pi) < 1e-7
        c0 = ~(c1 | c2)

        if not intrinsic:
            opt0_angles0 = half_sum - half_diff
            opt0_angles2 = half_sum + half_diff
        else:
            opt0_angles2 = half_sum - half_diff
            opt0_angles0 = half_sum + half_diff

        opt1_angles0 = 2 * half_sum
        opt2_angles0 = 2 * half_diff * (1 if intrinsic else -1)
        opt2_angles2 = xp.zeros_like(angles1)
        opt1_angles2 = xp.zeros_like(angles1)

        angles0 = xp.where(c0, opt0_angles0, xp.where(c1, opt1_angles0, opt2_angles0))
        angles2 = xp.where(c0, opt0_angles2, xp.where(c1, opt1_angles2, opt2_angles2))

        angles = xp.stack([angles0, angles1, angles2], axis=-1)
        if not symmetric:
            if not intrinsic:
                # These dtype and device arguments may be needed for e.g. GPU arrays
                angles *= xp.asarray([1.0, 1.0, sign])  # , dtype=angles.dtype, device=angles.device)
            else:
                angles *= xp.asarray([sign, 1.0, 1.0])  # , dtype=angles.dtype, device=angles.device)
            angles -= xp.asarray([0, xp.pi / 2, 0])  # , dtype=angles.dtype, device=angles.device)

        # Wrap to [-pi, pi]
        angles = (angles + xp.pi) % (2 * xp.pi) - xp.pi

        if degrees:
            angles = angles * 180 / xp.pi

        return angles

    def as_davenport(self, axes, order, degrees=False):
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def concatenate(cls, rotations: List[Self]):
        xp = array_api_compat.array_namespace(rotations[0]._quat)
        return cls(xp.concat([r._quat for r in rotations], axis=0))

    def apply(self, vectors, inverse=False):
        # Not sure if it is faster to calculate the rotation matrix and then apply it
        # Or use the quaternion
        # I guess I can test it.
        with_quats = False

        xp = array_api_compat.array_namespace(self._quat)

        if with_quats:
            other_q = xp.concat([vectors, xp.zeros_like(vectors[..., :1])], axis=-1)

            if inverse:
                v = self._quat_mul(self._quat_mul(self._quat_conj(self._quat), other_q), self._quat)
            else:
                v = self._quat_mul(self._quat_mul(self._quat, other_q), self._quat_conj(self._quat))

            return self._quat_imag_part(v)
        else:
            R = self.as_matrix()
            if inverse:
                return (xp.matrix_transpose(R) @ vectors[..., None])[..., 0]
            else:
                return (R @ vectors[..., None])[..., 0]

    def __mul__(self, other: Self):
        # Neccesary to normalize?
        return self.__class__(self._quat_mul(self._quat, other._quat), normalize=True)

    def __pow__(self, n: float, modulus=None):
        xp = array_api_compat.array_namespace(self._quat)
        if n == 0:
            return self.__class__.identity(xp, len(self))
        if n == -1:
            return self.inv()
        if n == 1:
            return self.__class__(self._quat)  # copy?
        else:
            return self.__class__.from_rotvec(n * self.as_rotvec())

    def inv(self):
        return self.__class__(self._quat_conj(self._quat))

    def magnitude(self):
        xp = array_api_compat.array_namespace(self._quat)
        norm123 = xp.linalg.vector_norm(self._quat[..., :3], axis=-1)
        return 2 * xp.atan2(norm123, xp.abs(self._quat[..., 3]))

    def approx_equal(self, other: Self, atol=None, degrees=False):
        raise NotImplementedError("Not implemented yet.")

    def mean(self, weights=None):
        # scipy copy pasta with some xp thrown in
        xp = array_api_compat.array_namespace(self._quat)
        if weights is None:
            weights = xp.ones(len(self), dtype=self._quat.dtype, device=self._quat.device)

        K = xp.matrix_transpose(weights[..., :, None] * self._quat) @ self._quat
        _, v = xp.linalg.eigh(K)
        return self.__class__(v[..., :, -1], normalize=False)

    def reduce(self, left=None, right=None, return_indices=False):
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def align_vectors(a, b, weights=None, return_sensitivity=False):
        raise NotImplementedError("Not implemented yet.")

    # For the three following methods, we have added the xp argument (along with dtype and device)
    # Since there have to be a way to create the array with the desired backend

    @classmethod
    def create_group(cls, xp, group, axis="Z", dtype=None, device=None):
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def random(cls, xp, shape=Optional[Union[int, Tuple]], random_state: Optional[Random] = None, dtype=None, device=None):
        # Normalizing random quaternions should give a uniform distribution of rotations
        if random_state is None:
            random_state = Random()
        if shape is None:
            shape = (4,)
        else:
            shape = (shape, 4)
        num = xp.prod(xp.asarray(shape))
        q = xp.reshape(xp.asarray([random_state.gauss(0, 1) for _ in range(num)], dtype=dtype, device=device), shape)
        return cls(q, normalize=True)

    @classmethod
    def identity(cls, xp, shape: Optional[Union[int, Tuple]] = None, dtype=None, device=None):
        if shape is None:
            shape = ()
        elif isinstance(shape, int):
            if shape == 1:
                shape = ()  # preserve single quaternion as (4,)
            else:
                shape = (shape,)
        return cls(xp.broadcast_to(xp.asarray([0, 0, 0, 1.0], dtype=dtype, device=device), shape + (4,)))
