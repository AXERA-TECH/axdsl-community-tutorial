"""Runnable AXDSL vector-add example used by the user documentation."""

import torch

import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN)
def vector_add_kernel(x, y, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])
    tile = slice(start, stop)

    x_tile = ax.dsl.load(x[tile])
    y_tile = ax.dsl.load(y[tile])
    result = ax.dsl.add(x_tile, y_tile)
    ax.dsl.store(x=result, r=output[tile])
    return output


def vector_add_wrapper(x, y, output):
    """Launch the vector-add kernel and return its output tensor."""
    TILE_SIZE = 128
    grid = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    vector_add_kernel[grid](x, y, output, TILE_SIZE)


def main():
    ax.init(chip_num=1, core_num=1)
    x = torch.arange(1000, dtype=torch.float32)
    y = torch.arange(1000, dtype=torch.float32) * 2
    assert x.shape == y.shape
    output_ax = torch.empty_like(x)

    output_gt = x + y
    vector_add_wrapper(x, y, output_ax)

    torch.testing.assert_close(output_gt, output_ax)
    print("AXDSL vector add: PASS")


if __name__ == "__main__":
    main()
