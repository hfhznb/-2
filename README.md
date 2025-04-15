import numpy as np
import matplotlib.pyplot as plt

# 参数定义
R1 = 8.3       # kpc
A = 20.41+-0.31      # kpc^(-2)
a = 9.03+-1.08       # 无单位
b = 13.99+-1.36      # 无单位
R_pdf = 3.76+-0.42   # kpc

# 密度函数
def rho(r):
    return A * ((r + R_pdf) / (R1 + R_pdf))**a * np.exp(-b * ((r - R1) / (R1 + R_pdf)))

# 提议分布 q(r)，这里选择均匀分布作为提议分布
def q(r):
    return 1 / (20 - 0)  # 假设 r 的范围是 [0, 20]

# 找到 M，使得 rho(r) <= M * q(r)
r_values = np.linspace(0, 20, 1000)  # 在 [0, 20] 范围内生成采样点
rho_values = rho(r_values)
M = np.max(rho_values) / q(r_values)  # 计算 M

# 拒绝采样
samples = []
num_samples = 130000  # 需要的样本数量
while len(samples) < num_samples:
    r_prime = np.random.uniform(0, 20)  # 从提议分布中抽样
    u = np.random.uniform(0, 1)         # 从 U(0, 1) 中抽样
    if u < rho(r_prime) / (M * q(r_prime)):  # 接受条件
        samples.append(r_prime)

# 绘制曲线和直方图
plt.figure(figsize=(12, 6))

# 曲线
plt.subplot(1, 2, 1)
plt.plot(r_values, rho_values, label='Density Function', color='blue')
plt.xlabel('r (kpc)')
plt.ylabel('ρ(r)')
plt.title('Density Function Curve')
plt.legend()

# 直方图
plt.subplot(1, 2, 2)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='green', label='Sample Distribution')
plt.xlabel('r (kpc)')
plt.ylabel('Frequency')
plt.title('Sample Distribution Histogram')
plt.legend()

plt.tight_layout()
plt.show()
