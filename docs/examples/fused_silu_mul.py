"""AXDSL fused SiLU-times-value tutorial example."""

import torch

import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN, enable_cache=False)
def fused_silu_mul_kernel(gate, value, output, tile_size):
    """Compute SiLU(gate) * value for one tile per program."""
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, gate.shape[0])

    gate_tile = ax.dsl.load(gate[start:stop])
    value_tile = ax.dsl.load(value[start:stop])

    activated = ax.dsl.silu(gate_tile)
    result = ax.dsl.mul(activated, value_tile)

    ax.dsl.store(x=result, r=output[start:stop])
    return output


def fused_silu_mul(gate, value, output, tile_size=128):
    grid = (gate.shape[0] + tile_size - 1) // tile_size
    fused_silu_mul_kernel[grid](gate, value, output, tile_size)
    return output


def reference(gate, value):
    return torch.nn.functional.silu(gate) * value


def main():
    torch.manual_seed(0)
    ax.init(chip_num=1, core_num=1)

    gate = torch.randn(1000, dtype=torch.bfloat16)
    value = torch.randn_like(gate)
    output = torch.empty_like(gate)

    fused_silu_mul(gate, value, output, tile_size=128)
    torch.testing.assert_close(reference(gate, value), output, atol=2e-2, rtol=2e-2)
    print("AXDSL fused SiLU*Mul: PASS")


if __name__ == "__main__":
    main()
