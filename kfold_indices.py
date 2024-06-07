import random
import os
filename_list = ['AB','AFIB','B','N','SBR','SVTA','T']
for filename in filename_list:
    with open(f'./data_indices/{filename}_indice.csv', 'r', encoding='utf-8') as file:
        # 读取所有的行，除了第一行
        lines = file.readlines()[1:]
        # 对其进行打乱 每次随机
        random.shuffle(lines)
        # 取80%的数据作为训练集，写入到train_kfold_indice.csv中
        trian_filename = './data_indices/train_kfold_indice.csv'
        with open(f'./data_indices/train_kfold_indice.csv', 'a', encoding='utf-8') as file:
            if os.path.getsize(trian_filename) == 0:
                file.write('label,Filename\n')
            for line in lines[:int(len(lines)*0.8)]:
                file.write(line)
            # 取20%的数据作为测试集，写入到test_kfold_indice.csv中
        test_filename = './data_indices/test_kfold_indice.csv'
        with open(f'./data_indices/test_kfold_indice.csv', 'a', encoding='utf-8') as file:
            if os.path.getsize(test_filename) == 0:
                file.write('label,Filename\n')
            for line in lines[int(len(lines)*0.8):]:
                file.write(line)