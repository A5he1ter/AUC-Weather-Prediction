import pandas as pd

# 读取 train.csv 文件
df_train = pd.read_csv('./data/weatherAUS.csv')

# 获取数据的行数
total_rows = len(df_train)

# 计算每个部分的大小
part_size = total_rows // 3

# 将数据分成三部分
part1 = df_train.iloc[:part_size]
part2 = df_train.iloc[part_size: 2 * part_size]
part3 = df_train.iloc[2 * part_size:]

# 将每个部分保存为新的 CSV 文件
part1.to_csv('./data/weatherAUS_0.csv', index=False)
part2.to_csv('./data/weatherAUS_1.csv', index=False)
part3.to_csv('./data/weatherAUS_2.csv', index=False)
