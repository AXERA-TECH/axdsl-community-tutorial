"""AXDSL tiled matmul tutorial example.

The first tutorial version assumes M and N are divisible by the tile sizes,
so it focuses on the 2D grid mapping and the DDR -> OCM -> DDR dataflow.
"""

import torch

import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def tiled_matmul_kernel(a, b, output, tile_size):
    """Compute output = a @ b.T with a 2D program grid."""
    tile_m, tile_n = tile_size

    pid_m = ax.program_id(0)
    pid_n = ax.program_id(1)

    m0 = pid_m * tile_m
    m1 = m0 + tile_m
    n0 = pid_n * tile_n
    n1 = n0 + tile_n

    # a: (M, K), b: (N, K)
    a_tile = ax.dsl.load(a[m0:m1, :])
    b_tile = ax.dsl.load(b[n0:n1, :])
    c_tile = ax.dsl.matmul(a_tile, b_tile, transposed=True)
    ax.dsl.store(x=c_tile, r=output[m0:m1, n0:n1])
    return output


def tiled_matmul(a, b, output, tile_m=128, tile_n=128):
    if a.shape[0] % tile_m or output.shape[1] % tile_n:
        raise ValueError("this tutorial requires M and N divisible by tile sizes")
    grid = (a.shape[0] // tile_m, output.shape[1] // tile_n)
    tiled_matmul_kernel[(grid, 1)](a, b, output, (tile_m, tile_n))
    return output


def main():
    torch.manual_seed(0)
    ax.init(chip_num=1, core_num=1)

    M, K, N = 256, 128, 256
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(N, K, dtype=torch.bfloat16)
    output = torch.empty(M, N, dtype=torch.bfloat16)

    reference = a @ b.transpose(0, 1)
    tiled_matmul(a, b, output, tile_m=64, tile_n=64)

    torch.testing.assert_close(reference, output, atol=2e-2, rtol=2e-2)
    print("AXDSL tiled matmul: PASS")


if __name__ == "__main__":
    main()
