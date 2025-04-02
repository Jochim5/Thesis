import numpy as np
import matplotlib.pyplot as plt

ga = [3.15, 7.66, 6.40, 3.52, 6.62, 4.40, 4.63, 3.59, 4.24, 6.85]
model = [8.07, 8.42, 6.63, 5.84, 9.54, 8.09, 7.85, 7.16, 5.74, 5.33]

# 计算平均值
mean1 = np.mean(ga)
mean2 = np.mean(model)

# 创建 x 轴坐标（1~10）
x = np.arange(1, 11)

# 创建画布
plt.figure(figsize=(10, 5))

# 画数据点（散点图）
plt.scatter(x, ga, color="blue", label="Fast GA (300 robots trained)", alpha=0.7)
plt.scatter(x, model, color="red", label="Discrete Diffusion Model", alpha=0.7)

# 画平均值柱状图 (x=5.5 对应 GA, x=6.5 对应 Model)
plt.bar(4.5, mean1, color="blue", alpha=0.6, width=0.6, label="GA Mean")
plt.bar(7.5, mean2, color="red", alpha=0.6, width=0.6, label="Model Mean")

# 添加数值标注
plt.text(4.5, mean1 + 0.2, f"{mean1:.2f}", ha="center", fontsize=12)
plt.text(7.5, mean2 + 0.2, f"{mean2:.2f}", ha="center", fontsize=12)

# 调整 x 轴刻度，使均值柱状图居中
plt.xticks(list(x) + [4.5, 7.5], labels=list(range(1, 11)) + ["GA Mean", "Model Mean"])

# 添加标签
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Comparison between GA and Model")
plt.legend()
plt.grid(True)

# 显示图表
plt.show()

# 打印平均值
print(f"Mean of GA: {mean1:.2f}")
print(f"Mean of Model: {mean2:.2f}")