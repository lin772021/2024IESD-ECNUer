import numpy as np
import pandas as pd
import os
import csv


# get features
def extract_features_for_single_peak_optimized_v1(x_peaks):
    peak_features_count = len(x_peaks)
    diff_intervals = [x - x_peaks[i - 1] for i, x in enumerate(x_peaks)][1:]

    if len(diff_intervals) > 1:
        peak_features_min_int = np.min(diff_intervals)
        peak_features_max_int = np.max(diff_intervals)
        peak_features_avg_int = np.ceil(np.average(diff_intervals))

    else:
        peak_features_min_int = 1
        peak_features_max_int = 1
        peak_features_avg_int = 1

    feature_list = [peak_features_count, peak_features_max_int, peak_features_min_int, peak_features_avg_int]
    return feature_list


def supress_non_maximum(peak_indices, X_data, window=30):
    if len(peak_indices) < 1:
        return []

    new_peak_indices = []
    last_peak = peak_indices[0]
    for i in range(1, len(peak_indices)):
        curr_diff = peak_indices[i] - last_peak
        if curr_diff > window:
            new_peak_indices.append(last_peak)
            last_peak = peak_indices[i]
        else:
            if X_data[peak_indices[i]] > X_data[last_peak]:
                last_peak = peak_indices[i]
    if len(new_peak_indices) == 0:
        return new_peak_indices
    if new_peak_indices[-1] != last_peak:
        new_peak_indices.append(last_peak)

    return new_peak_indices


def extract_peaks_features_optimized_v1(X_data, std_val=1.8, window=38):
    X_data_new = np.array(X_data)
    std_arr = np.abs(np.std(X_data_new) * std_val)
    peak_indices = np.where(np.abs(X_data_new) > std_arr)[0]

    peak_indices = supress_non_maximum(peak_indices, X_data, window)

    peaks_features = extract_features_for_single_peak_optimized_v1(peak_indices)

    return peaks_features


# def extract_features_extended(X_data, sigma=1.8, window=38):
#     X_data_peak_features = []
#     for i in range(len(X_data)):
#         X_data_features = extract_peaks_features_optimized_v1(X_data[i], sigma, 40)
#         X_data_peak_features.append(X_data_features)

#     X_data_peaks_feat_df = pd.DataFrame.from_dict(X_data_peak_features)

#     return X_data_peaks_feat_df


def extract_amplitude_features(X_data, index):
    # 使用两个值最大值和最小值
    X_data_peak_amplitude = []
    data = X_data[index]

    for i in range(len(data)):
        X_data_features = extract_peaks_features_optimized_v1(data[i], 1.8, 40)
        X_data_peak_amplitude.append([X_data_features[3], np.average(data[i]), np.max(data[i]), np.min(data[i])])

    return X_data_peak_amplitude

def extract_all_features(X_data):
    # 使用两个值最大值和最小值
    X_data_features = []
    X_data_features = extract_peaks_features_optimized_v1(X_data, 1.8, 40)
    X_data_features += [np.max(X_data), np.min(X_data), np.max(X_data) - np.min(X_data), np.average(X_data)]

    return X_data_features

def show_validation_results(C, total_time):
    # C = C_board
    print(C)

    # total_time = 0#sum(timeList)
    avg_time = total_time  # np.mean(timeList)
    acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
    precision = C[1][1] / (C[1][1] + C[0][1])
    sensitivity = C[1][1] / (C[1][1] + C[1][0])
    FP_rate = C[0][1] / (C[0][1] + C[0][0])
    PPV = C[1][1] / (C[1][1] + C[1][0])
    NPV = C[0][0] / (C[0][0] + C[0][1])
    F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    F_beta_score = (1 + 2 ** 2) * (precision * sensitivity) / ((2 ** 2) * precision + sensitivity)

    print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
          "\ntotal_time: {},\n average_time: {}".format(acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score,
                                                        total_time, avg_time))

    print("F_beta_score : ", F_beta_score)


# 加载原始数据集和标签
def loadData(root_dir, indice_dir, mode, size):
    print("Loading CSV data...")
    csvdata_all = loadCSV(os.path.join(indice_dir, mode + '_indice.csv'))

    data_list = []
    label_list = []
    filename_list = []
    for i, (k, v) in enumerate(csvdata_all.items()):
        text_path = root_dir + str(k)
        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, size)  # .reshape(self.size, 1)

        label = int(v[0])

        data_list.append(IEGM_seg)
        label_list.append(label)
        filename_list.append(str(k))

    data_array = np.array(data_list)

    return data_array, np.array(label_list), filename_list


def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

def loadPred(root_dir, name):
  label_list = []
  
  # 使用 with 语句打开文件
  filename = os.path.join(root_dir, name + '.txt')
  with open(filename, 'r') as file:
      # 逐行读取文件内容
    for line in file:
      label_list.append(int(line.strip()))
  return label_list



