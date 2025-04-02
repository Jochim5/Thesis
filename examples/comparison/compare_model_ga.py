import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  模型生成的 10 个奖励值
model_rewards = [0.811818182,0.94,0.854545455,0.897272727,0.917272727,0.897272727,0.854545455,0.854545455,0.726363636,0.854545455]

#  GA 算法在不同迭代次数下的 10 个奖励值
ga_rewards = {
    300: [0.532727273,0.660909091,0.658181818,0.552727273,0.595454545,0.552727273,0.638181818,0.680909091,0.680909091,0.638181818],
    500: [0.680909091,0.766363636,0.680909091,0.615454545,0.638181818,0.703636364,0.658181818,0.640909091,0.638181818,0.638181818],
    700: [0.766363636,0.766363636,0.766363636,0.680909091,0.595454545,0.660909091,0.700909091,0.766363636,0.786363636,0.595454545],
    900: [0.723636364,0.746363636,0.766363636,0.618181818,0.660909091,0.809090909,0.831818182,0.766363636,0.618181818,0.723636364],
    1100: [0.766363636,0.766363636,0.766363636,0.726363636,0.829090909,0.786363636,0.851818182,0.829090909,0.700909091,0.723636364],
    1300: [0.723636364,0.743636364,0.809090909,0.766363636,0.786363636,0.766363636,0.809090909,0.871818182,0.766363636,0.723636364],
    1500: [0.851818182,0.829090909,0.851818182,0.786363636,0.809090909,0.766363636,0.831818182,0.809090909,0.789090909,0.746363636]
}

# 计算平均值
model_avg = np.mean(model_rewards)
ga_avg = {iteration: np.mean(rewards) for iteration, rewards in ga_rewards.items()}

# 创建一个 DataFrame 表格
df_data = []
for iteration, rewards in ga_rewards.items():
    for reward in rewards:
        df_data.append([iteration, reward])

df = pd.DataFrame(df_data, columns=["robots trained", "reward"])

# 创建图表
plt.figure(figsize=(10, 6))

# 画 GA 的所有点（散点图）
for iteration, rewards in ga_rewards.items():
    plt.scatter([iteration] * len(rewards), rewards, color="blue", alpha=0.6, label="fast GA reward" if iteration == 300 else "")

# 画 GA 的平均奖励值（折线 + 点）
iterations = list(ga_avg.keys())
ga_means = list(ga_avg.values())
plt.plot(iterations, ga_means, marker="o", linestyle="-", color="red", label="fast GA average reward")

#  画模型的平均奖励值（水平直线）
plt.axhline(y=model_avg, color="green", linestyle="--", label=f"model average reward ({model_avg:.3f})")

# 添加标签
plt.xlabel("fast GA robot trained")
plt.ylabel("reward")
plt.title("diskrete diffusion model vs fast GA")
plt.legend()
plt.grid(True)

# 显示图表
plt.show()

# 显示数据表格
print(df)
