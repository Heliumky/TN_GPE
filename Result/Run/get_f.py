import re

# 读取文件内容
with open('nohup.out', 'r') as file:
    lines = file.readlines()

# 初始化一个列表来存储f的值
f_values = []
df_values = []

# 正则表达式匹配f的值
pattern = re.compile(r'f\s*=\s*([\d\.e\+\-]+)')
dfpattern = re.compile(r'df\s*=\s*([\d\.e\+\-]+)')

# 遍历每一行，找到f的值并添加到列表中
for line in lines:
    match = pattern.search(line)
    dfmatch = dfpattern.search(line)
    if match:
        f_value = match.group(1)
        f_values.append(f_value)
        df_value = dfmatch.group(1)
        df_values.append(df_value)

# 将f的值保存到新的文件
with open('f_values.txt', 'w') as output_file:
    for value in f_values:
        output_file.write(value + '\n')
        
with open('df_values.txt', 'w') as output_file:
    for value in df_values:
        output_file.write(value + '\n')

print(f"Extracted {len(f_values)} f values and saved to f_values.txt")

