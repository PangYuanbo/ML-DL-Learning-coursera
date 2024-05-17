import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 获取当前脚本所在的目录
directory = os.path.dirname(os.path.abspath(__file__))

# 列出当前目录下所有的CSV文件
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# 确保输出目录存在
output_directory = os.path.join(directory, 'plots')
os.makedirs(output_directory, exist_ok=True)

# 遍历每个CSV文件并绘制数据
for csv_file in csv_files:
    try:
        # 读取CSV文件
        data = pd.read_csv(csv_file)

        # 检查数据是否包含预期的列
        if 'Iteration' in data.columns and 'Cost' in data.columns:
            # 创建一个新的图表
            plt.figure(figsize=(10, 6))
            # 绘制数据
            plt.plot(data['Iteration'], data['Cost'])
            # 设置图表标题和标签
            plt.title(os.path.basename(csv_file))
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.grid(True)
            # 保存图表为文件
            plot_filename = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(csv_file))[0]}.png")
            plt.savefig(plot_filename)
            print(f"Plot saved as {plot_filename}")
            # 关闭图表以释放资源
            plt.close()
        else:
            print(f"File {csv_file} does not contain the expected columns.")
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
