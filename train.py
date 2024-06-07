import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import AFNet, RNN
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../data/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    args = argparser.parse_args()

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    path_net = args.path_net

    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='SVTA', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)

    ecg_signal = trainset[0]['ECG_seg'].numpy().reshape(-1)
    # ecg_signal
    print(ecg_signal)

    plt.plot(ecg_signal)
    plt.title('ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()