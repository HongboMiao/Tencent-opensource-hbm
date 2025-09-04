import asyncio
import numpy as np
import time

NUM_BLOCKS = 10000
BATCH_SIZE = 10000  # 每次发送的数据块数量
MAX_BLOCK_SIZE = 128 * (2 ** 12)  # 2MB

# 模拟缓冲区
max_buffer_size = NUM_BLOCKS * MAX_BLOCK_SIZE
send_buffer = np.empty(max_buffer_size, dtype=np.uint8)
recv_buffer = np.empty_like(send_buffer)

# 模拟发送函数（模拟网络传输）
async def sender(start_idx, end_idx, size):
    for i in range(start_idx, end_idx):
        offset = i * size
        # 模拟发送逻辑（例如：网络 IO 或 RDMA）
        await asyncio.sleep(0)  # 让出控制权模拟异步
        recv_buffer[offset:offset+size] = send_buffer[offset:offset+size]

# 主协程：模拟完整的多块数据传输
async def main():
    total_start = time.perf_counter()

    for exp in range(0, 12):  # 从128B到2MB
        size = 128 * (2 ** exp)
        print(f"\n===== 模拟传输 {NUM_BLOCKS} 个 {size}B 块 =====")

        # 填充发送缓冲区的模拟数据
        current_buffer_size = NUM_BLOCKS * size
        send_buffer[:current_buffer_size] = np.random.randint(
            0, 256, size=current_buffer_size, dtype=np.uint8
        )

        batch_start = time.perf_counter()

        tasks = []
        for i in range(0, NUM_BLOCKS, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, NUM_BLOCKS)
            tasks.append(sender(i, end_idx, size))

        await asyncio.gather(*tasks)

        batch_end = time.perf_counter()
        duration = batch_end - batch_start
        print(f"当前批次模拟完成，耗时: {duration:.4f} 秒")

        # 可选：验证数据一致性
        if not np.array_equal(send_buffer[:current_buffer_size], recv_buffer[:current_buffer_size]):
            print("❌ 数据验证失败！发送和接收内容不一致。")
        else:
            print("✅ 数据验证成功。")

    total_end = time.perf_counter()
    total_duration = total_end - total_start
    print(f"\n===== 所有模拟传输完成，总耗时: {total_duration:.4f} 秒 =====")

if __name__ == "__main__":
    asyncio.run(main())
