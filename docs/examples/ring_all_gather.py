from loguru import logger
import torch

from axnn.backend.base import MemoryType
from axnn.dsl import parallel as axd
import axnn.dsl as ax
from axnn.dsl.ctx import ctx
from axnn.dsl.util import try_resetting_seed


@ax.builder(run_mode=ax.RunMode.CRUN)
def ax_custom_op_kernel(sub_x_chip, output, signal, chip_id, CHIP_NUM):
    """
    shapes:
      - sub_x_chip: (M // CHIP_NUM, K)
      - output: (M, K)
    """
    m, K = sub_x_chip.shape  # m == M // CHIP_NUM
    M = m * CHIP_NUM

    # all_gather
    # step 0: 从 kernel 的输入读数, 发给右邻居
    # step 1-3: 从 shmem 读数, 发给右邻居
    # chip 为环形拓扑, 每次循环把一个数据块的发给右邻居
    right_chip_id = (chip_id + 1) % CHIP_NUM

    for step in range(CHIP_NUM):
        M_start = ((chip_id - step) * M // CHIP_NUM + M) % M
        # 向右邻居发数据
        if step == 0:
            src = sub_x_chip
        else:
            src = output[M_start : M_start + M // CHIP_NUM, :]
        dst = output[M_start : M_start + M // CHIP_NUM, :]
        axd.shmem_put(dst=dst, src=src, pe=right_chip_id)
        # update
        axd.shmem_signal_update(signal=signal, value=step + 1, pe=right_chip_id, sig_op=ax.SigOp.SET)
        # wait
        axd.signal_wait(signal=signal, value=step + 1)

    return output


if __name__ == "__main__":
    try_resetting_seed()

    M = 4
    K = 1024
    CHIP_NUM = 4
    x = torch.rand(size=(M, K), dtype=torch.float32)

    output_gt = torch.broadcast_to(x, (CHIP_NUM, M, K))
    # print(f"{output_gt=}")

    """
    launch kernel
    1. 开始时每个 chip 的 ddr 中有 x 的 1 / CHIP_NUM, shape 分别为 (M // CHIP_NUM, K)
    2. 经过 all_gather, 使每个 chip 都拥有完整的 x
    """
    ax.init(chip_num=CHIP_NUM, core_num=1)
    output_ax = torch.empty(size=(CHIP_NUM, M, K), dtype=torch.float32)
    # SHMEM 区域的 x 有 CHIP_NUM 个数据块, 而 signal 表示有几块数据已经 ready
    signal = ax.host.shmem_create_signal()
    for chip_id in range(CHIP_NUM):
        sub_x_chip = x[chip_id * M // CHIP_NUM : (chip_id + 1) * M // CHIP_NUM, :]  # M // CHIP_NUM, K
        ax_custom_op_kernel[1](sub_x_chip, output_ax[chip_id, :, :], signal, chip_id, CHIP_NUM)
    # print(f"{output_ax=}")

    torch.testing.assert_close(output_gt, output_ax, atol=1e-5, rtol=1e-5)
    logger.success("all close")
