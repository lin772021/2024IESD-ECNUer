import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import *
from tensorflow.keras.metrics import Precision, Recall
from tqdm import tqdm
import numpy as np
from help_code_demo_tf import FB
import random
from swa.tfkeras import SWA
from keras.callbacks import ModelCheckpoint

def main():
    seed = 365
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    path_net = args.path_net
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    # tf.config.experimental.set_visible_devices([], 'GPU')
    # print(tf.config.list_physical_devices('GPU'))

    # physical_devices = tf.config.list_physical_devices('CPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiating NN
    net = CNN_AF()
    # net = models.load_model(path_net + 'CNN_fb_1.h5')
    net.compile(
                optimizer=optimizers.Adam(learning_rate=LR),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    net.build(input_shape=(32, 1250, 1, 1))
    net.summary()

    # optimizer = optimizers.Adam(learning_rate=LR)
    # loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./saved_models/checkpoints/',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    swa = SWA(start_epoch=6, 
              lr_schedule='constant', # cyclic
              swa_lr=0.0001,
              swa_lr2=0.0005,
              swa_freq=5, 
              batch_size=BATCH_SIZE,
              verbose=1)

    # Start dataset loading
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)

    print("Start training")

    history = net.fit(trainloader, 
                      epochs=EPOCH, 
                      validation_data=testloader, 
                      verbose=1, 
                      callbacks=[model_checkpoint_callback, swa])
    net.save(f'./saved_models/CNN3_SWA.h5')

    # best_acc = 0.0
    # best_fb = 0.0
    # for epoch in range(EPOCH):
    #     running_loss = 0.0
    #     correct = 0.0
    #     accuracy = 0.0
    #     i = 0
    #     for step, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{EPOCH}", unit="batch")):
    #         with tf.GradientTape() as tape:
    #             # x = tf.reshape(x, (tf.size(x) / 1250, 1250, 1))
    #             logits = net(x, training=True)
    #             loss = loss_object(y, logits)
    #             grads = tape.gradient(loss, net.trainable_variables)
    #             optimizer.apply_gradients(zip(grads, net.trainable_variables))
    #             # swa_model.update_weights(optimizer.iterations.numpy())

    #             pred = tf.argmax(logits, axis=1)
    #             correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
    #             accuracy += correct / x.shape[0]
    #             correct = 0.0                
    #             running_loss += loss
    #             i += 1
    #     print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
    #           (epoch + 1, i, accuracy / i, running_loss / i))

    #     Train_loss.append(running_loss / i)
    #     Train_acc.append(accuracy / i)

    #     running_loss = 0.0
    #     accuracy = 0.0

    #     correct = 0.0
    #     total = 0.0
    #     i = 0.0
    #     running_loss_test = 0.0
    #     segs_TP = 0
    #     segs_TN = 0
    #     segs_FP = 0
    #     segs_FN = 0
    #     for x, y in tqdm(testloader, desc=f"Testing Epoch {epoch + 1}/{EPOCH}", unit="batch"):
    #         logits = net(x, training=False)
    #         test_loss = loss_object(y, logits)
    #         pred = tf.argmax(logits, axis=1)
    #         total += y.shape[0]
    #         correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))

    #         running_loss_test += test_loss
    #         i += x.shape[0]
    #         # predicted_test = tf.cast(predictions > 0.55, tf.int32)

    #         seg_label = y.numpy()[0]

    #         if seg_label == 0:
    #             segs_FP += np.sum(pred.numpy() != y.numpy())
    #             segs_TN += np.sum(pred.numpy() == y.numpy())
    #         elif seg_label == 1:
    #             segs_FN += np.sum(pred.numpy() != y.numpy())
    #             segs_TP += np.sum(pred.numpy() == y.numpy())

    #     fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)

    #     print('Test Acc: %.5f Test Loss: %.5f FB Score: %.5f' % (correct / total, running_loss_test / i, fb))

    #     Test_loss.append(running_loss_test / i)
    #     Test_acc.append((correct / total))
    #     # Save model
    #     if Test_acc[epoch] > best_acc:
    #         net.save(f'./saved_models/CNNAF_acc_{epoch+1}.h5')
    #         best_acc = Test_acc[epoch]
    #         print(f"=====\nBest acc model saved at Epoch: {epoch + 1}\n=====")
    #     if fb > best_fb:
    #         net.save(f'./saved_models/CNNAF_fb_{epoch+1}.h5')
    #         best_fb = fb
    #         print(f"=====\nBest fb model saved at Epoch: {epoch + 1}\n=====")


    # Write results to file
    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../data/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    args = argparser.parse_args()

    main()
