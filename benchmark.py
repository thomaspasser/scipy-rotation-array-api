import time
from typing import Any, List, Tuple

import cpuinfo
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.spatial.transform import Rotation as jRotation
from scipy.spatial.transform import Rotation as sRotation

from rotate import Rotation

cpu_info = cpuinfo.get_cpu_info()["brand_raw"]


TestCase = Tuple[str, sRotation | Rotation | jRotation, Any]


def run_benchmark(
    title: str,
    Ns: np.ndarray,
    test_matrix: List[TestCase],
    test_values: List[np.ndarray],
    n_repeat: int,
    func: Any,
    try_jit: bool = False,  # not currently working, jax.jit has issues with "array.device"
):
    results = {}
    for name, _, _ in test_matrix:
        results[name] = np.zeros_like(Ns, dtype=float)

    for i, N in enumerate(Ns):
        print(N)

        for name, tRotation, xp in test_matrix:
            print(name)

            if xp is jnp:
                if try_jit:

                    @jax.jit
                    def jitted_func(*args):
                        return func(tRotation)(*args)

                    def func2(*args):
                        return jitted_func(*args).block_until_ready()
                else:

                    def func2(*args):
                        return func(tRotation)(*args).block_until_ready()
            else:
                func2 = func(tRotation)

            test_values_xp = [xp.asarray(v[:N]) for v in test_values]

            res = []
            for _ in range(n_repeat):
                start = time.perf_counter()
                calc_res = func2(*test_values_xp)
                end = time.perf_counter()
                print(f"Time elapsed: {end - start:.6f} s")
                res.append(end - start)

            # average the times, without the slowest
            res.remove(max(res))
            results[name][i] = sum(res) / len(res)

    plt.figure()
    plt.title(f"{title}, CPU: {cpu_info}")

    for name in results:
        plt.plot(Ns, results[name], label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of rotations")
    plt.ylabel("Time elapsed (s)")
    plt.legend()
    plt.grid()


# quat to rot matrix


test_matrix: List[TestCase] = [
    ("sRotation_np", sRotation, np),
    ("jRotation_jnp", jRotation, jnp),
    ("this_np", Rotation, np),
    ("this_jnp", Rotation, jnp),
]

m = 6  # 7
Ns = np.logspace(1, m, m, base=10, dtype=int)

n_repeat = 3

# run_benchmark(
#     "quat -> matrix",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation: lambda q: tRotation.from_quat(q).as_matrix(),
#     try_jit=True,
# )

# run_benchmark(
#     "matrix -> quat",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_matrix()],
#     n_repeat,
#     lambda tRotation: lambda matrices: tRotation.from_matrix(matrices).as_quat(),
#     try_jit=True,
# )

# run_benchmark(
#     "quat mul",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_quat(), sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation: lambda q1, q2: (tRotation.from_quat(q1) * tRotation.from_quat(q2)).as_quat(),
#     try_jit=True,
# )

run_benchmark(
    "apply_N_rot_N_vec",
    Ns,
    test_matrix,
    [sRotation.random(Ns[-1]).as_quat(), np.random.normal(size=(Ns[-1], 3))],
    n_repeat,
    lambda tRotation: lambda q, v: tRotation.from_quat(q).apply(v),
    try_jit=True,
)

run_benchmark(
    "apply_N_rot_1_vec",
    Ns,
    test_matrix,
    [sRotation.random(Ns[-1]).as_quat(), np.random.normal(size=3)],
    n_repeat,
    lambda tRotation: lambda q, v: tRotation.from_quat(q).apply(v),
    try_jit=True,
)

run_benchmark(
    "apply_1_rot_N_vec",
    Ns,
    test_matrix,
    [sRotation.random(1).as_quat(), np.random.normal(size=(Ns[-1], 3))],
    n_repeat,
    lambda tRotation: lambda q, v: tRotation.from_quat(q).apply(v),
    try_jit=True,
)

# run_benchmark(
#     "quat -> euler",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation: lambda q: tRotation.from_quat(q).as_euler("xyz"),
#     try_jit=True,
# )

# run_benchmark(
#     "euler -> quat",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_euler("xyz")],
#     n_repeat,
#     lambda tRotation: lambda e: tRotation.from_euler("xyz", e).as_quat(),
#     try_jit=True,
# )

# run_benchmark(
#     "rotvec -> quat",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_rotvec()],
#     n_repeat,
#     lambda tRotation: lambda rv: tRotation.from_rotvec(rv).as_quat(),
#     try_jit=True,
# )

# run_benchmark(
#     "quat -> rotvec",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation: lambda q: tRotation.from_quat(q).as_rotvec(),
#     try_jit=True,
# )

# run_benchmark(
#     "quat inv",
#     Ns,
#     test_matrix,
#     [sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation: lambda q: tRotation.from_quat(q).inv().as_quat(),
#     try_jit=True,
# )

# without jRotation

# test_matrix2: List[TestCase] = [
#     ("scipy_numpy", sRotation, np),
#     ("this_numpy", Rotation, np),
#     ("this_jax", Rotation, jnp),
# ]

# run_benchmark(
#     "quat square",
#     Ns,
#     test_matrix2,
#     [sRotation.random(Ns[-1]).as_quat()],
#     n_repeat,
#     lambda tRotation, q: (tRotation.from_quat(q) ** 2).as_quat(),
# )

plt.show()
