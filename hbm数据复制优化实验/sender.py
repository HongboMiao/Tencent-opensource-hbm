#优化一 减少调用次数，将多个小block拼接成1个大buffer
#优化二 多个通道（多连接）提高带宽利用率
#优化三 动态调整批量大小，根据当前的传输状态动态调整 BATCH_SIZE
#优化四 预分配最大尺寸的缓冲区并复用


import ucp
import asyncio
import cupy as cp
import time

# ===================== 核心配置 =====================
NUM_BLOCKS = 10000
MAX_BLOCK_SIZE = 2 * 1024 * 1024
MAX_TOTAL_SIZE = NUM_BLOCKS * MAX_BLOCK_SIZE
NUM_CHANNELS = 6
RECEIVER_IP = "192.168.1.100"
BASE_PORT = 13337  # 基础端口，与接收端对应


# ===================== 动态批次管理器 =====================
class DynamicBatchManager:
    def __init__(self):
        self.current_batch = 500  # 初始批次大小
        self.best_duration = float('inf')

    def update_batch(self, current_duration):
        if current_duration < self.best_duration * 0.95:
            self.best_duration = current_duration

        is_high_load = current_duration > self.best_duration * 1.2
        if is_high_load:
            new_batch = max(self.current_batch - 200, 100)
            if new_batch != self.current_batch:
                print(f"[动态调整] 高负载！批次从{self.current_batch}→{new_batch}")
                self.current_batch = new_batch
        else:
            new_batch = min(self.current_batch + 200, 2000)
            if new_batch != self.current_batch:
                print(f"[动态调整] 负载正常！批次从{self.current_batch}→{new_batch}")
                self.current_batch = new_batch


# ===================== 单通道发送任务 =====================
async def send_channel(channel_id, block_range, batch_manager):
    """单通道发送任务（使用独立Endpoint）"""
    channel_start, channel_end = block_range
    port = BASE_PORT + channel_id
    print(f"[发送通道{channel_id}] 负责块范围：{channel_start}~{channel_end}，连接 {RECEIVER_IP}:{port}")

    # 建立独立连接
    ep = await ucp.create_endpoint(RECEIVER_IP, port)
    print(f"[发送通道{channel_id}] 连接建立：{ep}")

    # 预分配缓冲区
    meta_buf = cp.empty(3, dtype=cp.uint64)
    send_buf = None

    # 循环处理128B→2MB的所有块尺寸
    for exp in range(0, 15):
        block_size = 128 * (2** exp)
        current_batch_size = batch_manager.current_batch
        print(f"\n[发送通道{channel_id}] 开始传输：{NUM_BLOCKS}个{block_size}B块 | 当前批次：{current_batch_size}")

        # 生成数据（首次全量生成，后续仅覆盖必要部分）
        if send_buf is None:
            cp.random.seed(42 + channel_id)  # 通道间种子不同，避免数据重复
            send_buf = cp.random.randint(0, 255, size=MAX_TOTAL_SIZE, dtype=cp.uint8)
        else:
            send_buf[:NUM_BLOCKS * block_size] = cp.random.randint(0, 255, size=NUM_BLOCKS * block_size, dtype=cp.uint8)

        # 按批次发送
        for batch_start in range(channel_start, channel_end, current_batch_size):
            batch_end = min(batch_start + current_batch_size, channel_end)
            batch_block_count = batch_end - batch_start
            batch_byte_size = batch_block_count * block_size

            # 构建批次视图
            batch_view = send_buf[batch_start * MAX_BLOCK_SIZE : batch_start * MAX_BLOCK_SIZE + batch_byte_size]

            # 发送元数据
            meta_buf[0] = block_size
            meta_buf[1] = batch_start
            meta_buf[2] = batch_end
            await ep.send(meta_buf)

            # 发送数据
            batch_start_time = time.perf_counter()
            await ep.send(batch_view)

            # 发送校验和
            checksum = cp.sum(batch_view, dtype=cp.uint64)
            await ep.send(cp.array([checksum], dtype=cp.uint64))

            # 更新批次大小
            batch_duration = time.perf_counter() - batch_start_time
            print(f"[发送通道{channel_id}] 批次{batch_start}~{batch_end}：耗时{batch_duration:.4f}s | 校验和：{checksum}")
            batch_manager.update_batch(batch_duration)

    # 关闭当前通道连接
    await ep.close()
    print(f"[发送通道{channel_id}] 连接已关闭")


# ===================== 主函数 =====================
async def main():
    ucp.init()

    # 拆分块到多通道
    blocks_per_channel = NUM_BLOCKS // NUM_CHANNELS
    channel_blocks = []
    for i in range(NUM_CHANNELS):
        chan_start = i * blocks_per_channel
        chan_end = (i + 1) * blocks_per_channel if i < NUM_CHANNELS - 1 else NUM_BLOCKS
        channel_blocks.append((chan_start, chan_end))

    # 初始化动态批次管理器（每个通道独立管理）
    batch_managers = [DynamicBatchManager() for _ in range(NUM_CHANNELS)]

    # 启动多通道并发发送（每个通道独立连接）
    tasks = [
        asyncio.create_task(send_channel(i, channel_blocks[i], batch_managers[i]))
        for i in range(NUM_CHANNELS)
    ]
    await asyncio.gather(*tasks)

    print("\n✅ 所有通道传输完成")


if __name__ == "__main__":
    asyncio.run(main())