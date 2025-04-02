import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np

# 定义基础路径
base_dir = r"C:\d_pan\PythonProject\pythonProject\pythonProject\evogym\examples\saved_data\hand_design_experiment"

# 输入 x 的值
x = int(input("请输入 x 的值（最大代数）："))

# 初始化一个空列表，用于存储所有文件的第二列数据
all_data = []

# 遍历从 generation_0 到 generation_x 的文件夹
for generation in range(x + 1):
    # 构建文件夹路径
    generation_dir = os.path.join(base_dir, f"generation_{generation}")

    # 构建 output.txt 文件路径
    output_file_path = os.path.join(generation_dir, "output.txt")

    # 检查文件是否存在
    if not os.path.exists(output_file_path):
        print(f"警告：{output_file_path} 不存在，跳过该文件。")
        continue

    # 读取文件并提取第二列数据
    with open(output_file_path, "r") as file:
        lines = file.readlines()  # 读取文件的所有行
        data = [float(line.split()[1]) for line in lines]  # 提取第二列数据
        all_data.append(data)  # 将数据添加到列表中

# 将列表转换为 numpy 矩阵
reward_data = np.array(all_data)

# 计算平均奖励和最大奖励
generations = range(1, len(reward_data) + 1)
average_rewards = [np.mean(gen) for gen in reward_data]
max_rewards = [np.max(gen) for gen in reward_data]

# 绘制图形
plt.figure(figsize=(8, 6))

# 绘制每代的每个机器人的奖励值点
for gen_idx, rewards in enumerate(reward_data):
    plt.scatter([gen_idx + 1] * len(rewards), rewards, color='blue', alpha=0.7, label='Individual Rewards' if gen_idx == 0 else "")

# 绘制平均奖励曲线
plt.plot(generations, average_rewards, color='green', marker='o', label='Average Reward')

# 绘制最大奖励曲线
plt.plot(generations, max_rewards, color='red', linestyle='--', marker='s', label='Max Reward')

# 设置图例和标签
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Reward Progress Over Generations', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()