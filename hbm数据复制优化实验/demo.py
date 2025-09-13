#优化一 减少调用次数，将多个小block拼接成1个大buffer
#优化二 多个通道（多连接）提高带宽利用率
#优化三 动态调整批量大小，根据当前的传输状态动态调整 BATCH_SIZE
#优化四 预分配最大尺寸的缓冲区并复用

import asyncio
import numpy as np
import time


NUM_BLOCKS = 10000
INITIAL_BATCH_SIZE = 9000  # 初始批次大小
MIN_BATCH_SIZE = 500       # 最小批次大小
MAX_BATCH_SIZE = 20000      # 最大批次大小
LOAD_THRESHOLD = 1.2        # 高负载阈值
ADJUST_STEP = 1000          # 批次调整步长
MAX_BLOCK_SIZE = 128 * (2 ** 11)  # 单个块最大尺寸（2MB/8）
NUM_CHANNELS = 8            # 多通道数量
# 计算最大缓冲区尺寸
MAX_TOTAL_BUFFER_SIZE = NUM_BLOCKS * MAX_BLOCK_SIZE


class DynamicBatchManager:
    def __init__(self):
        self.current_batch = INITIAL_BATCH_SIZE  # 当前批次大小
        self.best_duration = float('inf')        

    def update_batch(self, current_duration):

        if current_duration < self.best_duration * 0.95:
            self.best_duration = current_duration
        is_high_load = current_duration > self.best_duration * LOAD_THRESHOLD

        if is_high_load:
            new_batch = self.current_batch - ADJUST_STEP
            self.current_batch = max(new_batch, MIN_BATCH_SIZE)  
            print(f"负载！批次大小从{self.current_batch + ADJUST_STEP}调整为{self.current_batch}")
        else:
            new_batch = self.current_batch + ADJUST_STEP
            if new_batch <= MAX_BATCH_SIZE:  
                self.current_batch = new_batch
                print(f"负载正常！批次大小从{self.current_batch - ADJUST_STEP}调整为{self.current_batch}")

# 预分配最大尺寸缓冲区
# 1. 发送缓冲区
send_buffer = np.empty(MAX_TOTAL_BUFFER_SIZE, dtype=np.uint8)
# 2. 接收缓冲区
recv_buffer = np.empty_like(send_buffer)
print(f"✅ 预分配完成：发送/接收缓冲区各 {MAX_TOTAL_BUFFER_SIZE/(1024**2):.2f} MB（最大尺寸）")

# 模拟发送函数
async def sender(channel_id, start_idx, end_idx, current_block_size):
    batch_byte_start = start_idx * current_block_size  
    batch_byte_end = end_idx * current_block_size      
    
    await asyncio.sleep(0)  # 模拟异步IO让出控制权
    recv_buffer[batch_byte_start:batch_byte_end] = send_buffer[batch_byte_start:batch_byte_end]

# 主协程：模拟完整的多块数据传输
async def main():
    total_start = time.perf_counter()
    batch_manager = DynamicBatchManager()

    for exp in range(0, 12):
        current_block_size = 128 * (2 ** exp)
        current_total_bytes = NUM_BLOCKS * current_block_size

        print(f"\n===== 模拟传输 {NUM_BLOCKS} 个 {current_block_size}B 块 | {NUM_CHANNELS}个通道 | 当前批次:{batch_manager.current_batch} =====")

        # 缓冲区复用
        send_buffer[:current_total_bytes] = np.random.randint(
            0, 256, size=current_total_bytes, dtype=np.uint8
        )

        batch_start = time.perf_counter()
        tasks = []

        # 多通道任务拆分
        blocks_per_channel = NUM_BLOCKS // NUM_CHANNELS
        for channel_id in range(NUM_CHANNELS):
            channel_block_start = channel_id * blocks_per_channel
            # 最后一个通道承担剩余所有块，避免数据遗漏
            channel_block_end = (channel_id + 1) * blocks_per_channel if channel_id < NUM_CHANNELS - 1 else NUM_BLOCKS
            
            # 按动态批次大小拆分任务
            for block_idx in range(channel_block_start, channel_block_end, batch_manager.current_batch):
                batch_block_end = min(block_idx + batch_manager.current_batch, channel_block_end)
                tasks.append(sender(channel_id, block_idx, batch_block_end, current_block_size))

        # 并发执行所有通道任务
        await asyncio.gather(*tasks)

        # 耗时统计与数据验证
        batch_end = time.perf_counter()
        duration = batch_end - batch_start
        print(f"当前批次耗时: {duration:.4f} 秒 ")
        batch_manager.update_batch(duration)

        # # 数据验证
        # if not np.array_equal(send_buffer[:current_total_bytes], recv_buffer[:current_total_bytes]):
        #     print("❌ 数据验证失败！发送和接收内容不一致。")
        # else:
        #     print("✅ 数据验证成功。")

    # total_end = time.perf_counter()
    # total_duration = total_end - total_start
    # print(f"\n===== 所有模拟传输完成，总耗时: {total_duration:.4f} 秒 =====")

if __name__ == "__main__":
    asyncio.run(main())