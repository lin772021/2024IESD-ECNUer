import os

def classify_indice(src):
    # 使用'open'函数以读取模式打开文件
    with open(f'./data_indices/{src}', 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            if len(line.split('-')) < 2: # 跳过第一行
                continue
            label = line.split('-')[1]
            filename = f'./data_indices/{label}_indice.csv'
            
            with open(filename, 'a', encoding='utf-8') as file:
                if os.path.getsize(filename) == 0:
                    file.write('label,Filename\n')
                file.write(line)
    return

if __name__ == '__main__':
    classify_indice('train_indice.csv')
    classify_indice('test_indice.csv')