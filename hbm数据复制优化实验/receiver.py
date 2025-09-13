import ucp
import asyncio
import cupy as cp
import time

# ===================== 核心配置 =====================
NUM_BLOCKS = 10000
MAX_BLOCK_SIZE = 2 * 1024 * 1024
MAX_TOTAL_SIZE = NUM_BLOCKS * MAX_BLOCK_SIZE
NUM_CHANNELS = 6
BASE_PORT = 13337  # 基础端口，每个通道使用BASE_PORT + channel_id


# ===================== 单通道接收任务 =====================
async def recv_channel(ep, channel_id, block_range):
    """单通道接收任务（使用独立Endpoint）"""
    channel_start, channel_end = block_range
    print(f"[接收通道{channel_id}] 负责块范围：{channel_start}~{channel_end}，连接：{ep}")

    # 预分配缓冲区
    recv_buf = cp.empty(MAX_TOTAL_SIZE, dtype=cp.uint8)
    meta_buf = cp.empty(3, dtype=cp.uint64)  # [block_size, batch_start, batch_end]
    checksum_buf = cp.empty(1, dtype=cp.uint64)

    # 循环处理128B→2MB的所有块尺寸
    for exp in range(0, 15):
        block_size = 128 * (2 **exp)
        print(f"\n[接收通道{channel_id}] 准备接收：{NUM_BLOCKS}个{block_size}B块")

        batch_idx = 0
        while batch_idx < NUM_BLOCKS:
            # 接收元数据
            await ep.recv(meta_buf)
            recv_block_size = int(meta_buf[0])
            batch_start = int(meta_buf[1])
            batch_end = int(meta_buf[2])
            batch_block_count = batch_end - batch_start
            batch_byte_size = batch_block_count * recv_block_size

            # 过滤非本通道的批次
            if not (channel_start <= batch_start < channel_end):
                batch_idx = batch_end
                continue

            # 接收数据
            batch_view = recv_buf[batch_start * MAX_BLOCK_SIZE : batch_start * MAX_BLOCK_SIZE + batch_byte_size]
            recv_start_time = time.perf_counter()
            await ep.recv(batch_view)

            # 验证校验和
            await ep.recv(checksum_buf)
            recv_checksum = cp.sum(batch_view, dtype=cp.uint64)
            send_checksum = checksum_buf[0]

            # 输出结果
            recv_duration = time.perf_counter() - recv_start_time
            if recv_checksum == send_checksum:
                print(f"[接收通道{channel_id}] 批次{batch_start}~{batch_end}：验证通过 | 耗时{recv_duration:.4f}s")
            else:
                print(f"[接收通道{channel_id}] 批次{batch_start}~{batch_end}：验证失败！发送{send_checksum}≠接收{recv_checksum}")

            batch_idx = batch_end

    await ep.close()
    print(f"[接收通道{channel_id}] 连接已关闭")


# ===================== 通道监听函数 =====================
async def channel_listener(channel_id, block_range):
    """为每个通道创建独立监听器"""
    port = BASE_PORT + channel_id

    async def handler(ep):
        """处理当前通道的连接"""
        await recv_channel(ep, channel_id, block_range)

    listener = ucp.create_listener(handler, port=port)
    print(f"[接收通道{channel_id}] 已启动，监听端口 {port}")
    return listener


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

    # 为每个通道启动独立监听器
    listeners = [
        await channel_listener(i, channel_blocks[i])
        for i in range(NUM_CHANNELS)
    ]

    print(f"✅ 接收端所有通道启动完成，基础端口 {BASE_PORT}")

    # 保持监听
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())