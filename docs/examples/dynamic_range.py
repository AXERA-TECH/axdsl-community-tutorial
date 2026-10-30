from loguru import logger
import numpy as np
import torch

import axnn.dsl as ax

TOTAL_TILES = 10
UNROLL_FACTOR = 2
START = 0
STOP = TOTAL_TILES
STEP = 1
TILE_M = 16
TILE_N = 16


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def ax_custom_op_kernel(a, output):
    index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)
    range_start = ax.dsl.from_numpy(np.array([START], dtype=np.int32), prefix="range_start")
    range_stop = ax.dsl.from_numpy(np.array([STOP], dtype=np.int32), prefix="range_stop")
    range_step = ax.dsl.from_numpy(np.array([STEP], dtype=np.int32), prefix="range_step")
    zero_offset = ax.dsl.from_numpy(np.zeros(shape=(1,), dtype=np.uint32), prefix="for_zero_offset")
    tile_shape = ax.dsl.from_numpy(np.ones(shape=(1,), dtype=np.uint32), prefix="for_tile_shape")

    def body_func(range_index, a, output, tile_shape, zero_offset):
        read_offset = ax.dsl.cast(range_index, dtype=ax.uint32)
        a_tile = ax.dsl.dynamic_ld(
            a,
            read_offset,
            tile_shape,
            dim=(0,),
            output_shape=(1, TILE_M, TILE_N),
        )
        a_tile = a_tile.reshape(TILE_M, TILE_N)

        add_tmp = ax.dsl.add(a_tile, a_tile)
        sub_tmp = ax.dsl.sub(add_tmp, a_tile)
        matmul_tmp = ax.dsl.matmul(sub_tmp, a_tile, transposed=True)
        mul_tmp = ax.dsl.mul(matmul_tmp, sub_tmp, output_dtype=a_tile.dtype)

        ax.dsl.dynamic_st(
            mul_tmp.reshape(1, TILE_M, TILE_N),
            zero_offset,
            read_offset,
            tile_shape,
            dim=(0,),
            output_shape=output.shape,
            r=output,
        )
        return a, output, tile_shape, zero_offset

    ax.dsl.range(
        index,
        range_start,
        range_stop,
        range_step,
        body_func,
        a,
        output,
        tile_shape,
        zero_offset,
        unroll_factor=UNROLL_FACTOR,
    )

    return output


def custom_op_run(a, output):
    return ax_custom_op_kernel[1](a, output)


def torch_reference(a):
    output = torch.zeros_like(a)
    for tile_idx in range(START, STOP, STEP):
        a_tile = a[tile_idx]
        add_tmp = a_tile + a_tile
        sub_tmp = add_tmp - a_tile
        matmul_tmp = sub_tmp @ a_tile.transpose(-1, -2)
        output[tile_idx] = matmul_tmp * sub_tmp
    return output


def test_demo_range():
    ax.init(chip_num=1, core_num=1)

    torch.manual_seed(0)
    a = (torch.rand((TOTAL_TILES, TILE_M, TILE_N), dtype=torch.bfloat16) * 0.125).to(torch.bfloat16)
    output_torch = torch_reference(a)
    output_ax = torch.zeros_like(output_torch)

    custom_op_run(a, output_ax)

    torch.testing.assert_close(output_ax, output_torch, atol=1e-2, rtol=1e-2)
    logger.success("range demo passed")


if __name__ == "__main__":
    test_demo_range()
