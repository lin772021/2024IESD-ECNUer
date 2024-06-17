import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import models
import tensorflow as tf
from help_code import ECG_DataSET, ToTensor, create_dataset, F1, FB, Sensitivity, Specificity, BAC, ACC, PPV, NPV
from models.best_model import *
import sys


def get_subjects(path_indices):
    test_indice_path = path_indices + 'test_indice.csv'
    test_indices = pd.read_csv(test_indice_path)
    subjects = test_indices['Filename'].apply(lambda x: x.split('-')[0]).unique().tolist()
    return subjects


def get_subjects_dataset(args):
    # Hyperparameters
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    subjects = get_subjects(path_indices)
    subjectloaders = []

    for subject_id in subjects:
        testset = ECG_DataSET(root_dir=path_data,
                              indice_dir=path_indices,
                              mode='test',
                              size=SIZE,
                              subject_id=subject_id,
                              transform=ToTensor())

        testloader = create_dataset(testset, BATCH_SIZE_TEST)
        subjectloaders.append(testloader)

    return subjectloaders


def test_by_subject(net, subjectloaders):
    # List to store metrics for each participant
    subject_metrics = []
    subjects_above_threshold = 0
    sid = 0

    for testloader in subjectloaders:
        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        # for x, y in tqdm(testloader, desc=f"Testing Epoch {epoch + 1}/{EPOCH}", unit="batch"):
        for ECG_test, labels_test in tqdm(testloader, desc=f"Testing Subject {sid}/{len(subjectloaders)}", unit="batch"):
            predictions = net(ECG_test, training=False)
            predicted_test = tf.argmax(predictions, axis=1)

            seg_label = labels_test.numpy()[0]

            if seg_label == 0:
                segs_FP += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TN += np.sum(predicted_test.numpy() == labels_test.numpy())
            elif seg_label == 1:
                segs_FN += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TP += np.sum(predicted_test.numpy() == labels_test.numpy())

        # Calculate metrics for the current participant
        fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)

        subject_metrics.append([fb])
        if fb > 0.9:
            subjects_above_threshold += 1
        
        sid += 1

    subject_metrics_array = np.array(subject_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    avg_fb = average_metrics[0]

    # Print average metric values
    print(f"Final F-B: {avg_fb:.5f}")

    proportion_above_threshold = subjects_above_threshold / len(subjectloaders)
    print("G Score: ", proportion_above_threshold)

    return avg_fb, proportion_above_threshold


def main():
    # Hyperparameters
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    path_data = args.path_data
    path_records = args.path_record
    path_net = args.path_net
    path_indices = args.path_indices
    model_name = args.model_name

    # List of subject_id
    subjects = get_subjects(path_indices)

    # List to store metrics for each participant
    subject_metrics = []

    # Load trained network
    net = models.load_model(path_net + f'{model_name}.h5')
    net.summary()
    
    subjects_above_threshold = 0

    print(subjects)
    for subject_id in subjects:
        print(f'[INFO] subject_id: {subject_id}')
        testset = ECG_DataSET(root_dir=path_data,
                               indice_dir=path_indices,
                               mode='test',
                               size=SIZE,
                               subject_id=subject_id,
                               transform=ToTensor())

        testloader = create_dataset(testset, BATCH_SIZE_TEST)

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        i = 0
        for ECG_test, labels_test in testloader:
            predictions = net(ECG_test, training=False)
            predicted_test = tf.argmax(predictions, axis=1)
            print(predicted_test.numpy()[0])

            seg_label = labels_test.numpy()[0]

            if seg_label == 0:
                segs_FP += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TN += np.sum(predicted_test.numpy() == labels_test.numpy())
            elif seg_label == 1:
                segs_FN += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TP += np.sum(predicted_test.numpy() == labels_test.numpy())

        # Calculate metrics for the current participant
        f1 = round(F1([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        se = round(Sensitivity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        sp = round(Specificity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        bac = round(BAC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        acc = round(ACC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        ppv = round(PPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        npv = round(NPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)

        subject_metrics.append([f1, fb, se, sp, bac, acc, ppv, npv])
        if fb > 0.9:
            subjects_above_threshold += 1

    subject_metrics_array = np.array(subject_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    avg_f1, avg_fb, avg_se, avg_sp, avg_bac, avg_acc, avg_ppv, avg_npv = average_metrics

    # Print average metric values
    print(f"Final F-1: {avg_f1:.5f}")
    print(f"Final F-B: {avg_fb:.5f}")
    print(f"Final SEN: {avg_se:.5f}")
    print(f"Final SPE: {avg_sp:.5f}")
    print(f"Final BAC: {avg_bac:.5f}")
    print(f"Final ACC: {avg_acc:.5f}")
    print(f"Final PPV: {avg_ppv:.5f}")
    print(f"Final NPV: {avg_npv:.5f}")

    proportion_above_threshold = subjects_above_threshold / len(subjects)
    print("G Score:", proportion_above_threshold)

    with open(path_records + 'seg_stat.txt', 'w') as f:
        f.write(f"Final F-1: {avg_f1:.5f}\n")
        f.write(f"Final F-B: {avg_fb:.5f}\n")
        f.write(f"Final SEN: {avg_se:.5f}\n")
        f.write(f"Final SPE: {avg_sp:.5f}\n")
        f.write(f"Final BAC: {avg_bac:.5f}\n")
        f.write(f"Final ACC: {avg_acc:.5f}\n")
        f.write(f"Final PPV: {avg_ppv:.5f}\n")
        f.write(f"Final NPV: {avg_npv:.5f}\n\n")
        f.write(f"G Score: {proportion_above_threshold}\n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../../data/training_dataset/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    argparser.add_argument('--path_record', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--model_name', type=str, default='CNN_LSTM_acc_16')
    args = argparser.parse_args()

    with open(f'{args.path_record}{args.model_name}_pred.txt', 'w') as f:
        stdout = sys.stdout
        sys.stdout = f
        
        main()

        sys.stdout = stdout
