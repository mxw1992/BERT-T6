import pandas as pd
import glob

# 获取所有CSV文件路径（假设都在当前目录下，命名为data1.csv到data16.csv）
csv_files = sorted(glob.glob('*last_layer_attention*.csv'))[:16]  # 只取前16个匹配的文件

# 检查是否找到16个文件
if len(csv_files) != 16:
    raise ValueError(f"找到 {len(csv_files)} 个CSV文件，但需要16个")

# 初始化一个DataFrame用于存储总和（以第一个文件的结构为基准）
sum_df = pd.read_csv(csv_files[0]).astype(float) * 0  # 创建全0的DataFrame

# 遍历所有CSV文件并累加
for file in csv_files:
    df = pd.read_csv(file)
    # 确保所有DataFrame形状相同
    if df.shape != sum_df.shape:
        raise ValueError(f"文件 {file} 的形状 {df.shape} 不匹配基准形状 {sum_df.shape}")
    sum_df += df.astype(float)  # 转换为float后相加

# 保存结果
sum_df.to_csv('sum_result.csv', index=False)
print("所有CSV文件已按位置相加，结果保存为 sum_result.csv")

# 显示前几行结果（可选）
print("\n结果预览：")
print(sum_df.head())