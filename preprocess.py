import os

# 使用'open'函数以读取模式打开文件
with open('./data_indices/train_indice.csv', 'r', encoding='utf-8') as file:
    # 逐行读取文件
    for line in file:
        # print(line)

        if len(line.split('-')) < 2: # 跳过第一行
            continue
        label = line.split('-')[1]
        # print(label)

        filename = f'./data_indices/{label}_indice.csv'
        
        with open(filename, 'a', encoding='utf-8') as file:
            if os.path.getsize(filename) == 0:
                file.write('label,Filename\n')
            file.write(line)

with open('./data_indices/test_indice.csv', 'r', encoding='utf-8') as file:
    # 逐行读取文件
    for line in file:
        # print(line)

        if len(line.split('-')) < 2: # 跳过第一行
            continue
        label = line.split('-')[1]
        # print(label)

        filename = f'./data_indices/{label}_indice.csv'
        
        with open(filename, 'a', encoding='utf-8') as file:
            if os.path.getsize(filename) == 0:
                file.write('label,Filename\n')
            file.write(line)