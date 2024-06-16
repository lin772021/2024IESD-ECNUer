import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from tqdm import tqdm
import numpy as np
import random
from swa.tfkeras import SWA

from torch.utils.tensorboard import SummaryWriter

from help_code import *
from models.best_model import *
from models.RNN import *
from models.LSTM import *
from test import *


def auto_train_swa(args, net, trainloader, testloader):
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    path_net = args.path_net

    net.compile(optimizer=optimizers.Adam(learning_rate=LR),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{path_net}checkpoints/',
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

    history = net.fit(trainloader, 
                      epochs=EPOCH, 
                      validation_data=testloader, 
                      verbose=1, 
                      callbacks=[model_checkpoint_callback, swa])
    net.save(f'{path_net}{net.name}.h5')
    return


def main():
    # Hyperparameters
    seed = args.seed
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    path_net = args.path_net

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 
    # Instantiating NN
    # 
    # nets = [CNN_AF(seed), CNN_RNN1(seed), CNN_RNN2(seed), CNN_RNN3(seed), CNN_LSTM(seed)]
    nets = [CNN_LSTM(seed)]
    # net = models.load_model(path_net + '.h5')
    # net.build(input_shape=(32, 1250, 1, 1))
    # net.summary()

    # 
    # Create datasets
    # 
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='all', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)

    subjectloaders = get_subjects_dataset(args)

    for net in nets:
        Train_loss = []
        Train_acc = []
        Test_loss = []
        Test_acc = []

        #
        # 1. Automatic training with Stochastic Weight Averaging (SWA)
        # 
        # auto_train_swa(args, net, trainloader, testloader)

        # 
        # 2. Customized training
        # 
        optimizer = optimizers.Adam(learning_rate=LR)
        loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

        best_acc = 0.0 # Save the model with the best accuracy
        best_fb = 0.0  # Save the model with the best F_beta

        # Set TensorBoard logdir
        writer = SummaryWriter(f'./runs/{net.name}_all')
        
        for epoch in range(EPOCH):
            running_loss = 0.0
            accuracy = 0.0
            correct = 0.0
            i = 0
            
            # 
            # Train
            # 
            for step, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{EPOCH}", unit="batch")):
                with tf.GradientTape() as tape:
                    logits = net(x, training=True)
                    loss = loss_object(y, logits)
                    grads = tape.gradient(loss, net.trainable_variables)
                    optimizer.apply_gradients(zip(grads, net.trainable_variables))

                    pred = tf.argmax(logits, axis=1)
                    correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                    accuracy += correct / x.shape[0]
                    correct = 0.0                
                    running_loss += loss
                    i += 1
            print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
                (epoch + 1, i, accuracy / i, running_loss / i))

            Train_loss.append(running_loss / i)
            Train_acc.append(accuracy / i)

            # Save training loss and acc to TensorBoard
            writer.add_scalar('Loss/Train', np.array(Train_loss[epoch]), epoch + 1)
            writer.add_scalar('Accuracy/Train', np.array(Train_acc[epoch]), epoch + 1)

            correct = 0.0
            total = 0.0
            i = 0.0
            running_loss_test = 0.0

            segs_TP = 0
            segs_TN = 0
            segs_FP = 0
            segs_FN = 0
            
            # 
            # Test
            # 
            for x, y in tqdm(testloader, desc=f"Testing Epoch {epoch + 1}/{EPOCH}", unit="batch"):
                logits = net(x, training=False)
                test_loss = loss_object(y, logits)

                pred = tf.argmax(logits, axis=1)
                total += y.shape[0]
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))

                running_loss_test += test_loss
                i += x.shape[0]

                seg_label = y.numpy()[0]

                if seg_label == 0:
                    segs_FP += np.sum(pred.numpy() != y.numpy())
                    segs_TN += np.sum(pred.numpy() == y.numpy())
                elif seg_label == 1:
                    segs_FN += np.sum(pred.numpy() != y.numpy())
                    segs_TP += np.sum(pred.numpy() == y.numpy())
            fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
            print('Test Acc: %.5f Test Loss: %.5f FB Score: %.5f' % (correct / total, running_loss_test / i, fb))

            Test_loss.append(running_loss_test / i)
            Test_acc.append((correct / total))

            writer.add_scalar('Loss/Test', np.array(Test_loss[epoch]), epoch + 1)
            writer.add_scalar('Accuracy/Test', np.array(Test_acc[epoch]), epoch + 1)
            writer.add_scalar('F-B/Test', np.array(fb), epoch + 1)

            # Save model
            if Test_acc[epoch] > best_acc:
                net.save(f'{path_net}{net.name}_acc.h5')
                best_acc = Test_acc[epoch]
                print(f"=====\n[INFO] Best acc model saved at Epoch: {epoch + 1}")
            if fb > best_fb:
                net.save(f'{path_net}{net.name}_fb.h5')
                best_fb = fb
                print(f"=====\n[INFO] Best fb model saved at Epoch: {epoch + 1}")

            # 
            # Test by subject
            # 
            # avg_fb, G_score = test_by_subject(net, subjectloaders)
            # writer.add_scalar('Final F-B/Test by Subject', np.array(avg_fb), epoch + 1)
            # writer.add_scalar('G Score/Test by Subject', np.array(G_score), epoch + 1)

        net.save(f'{path_net}{net.name}_{EPOCH}.h5')
        
        # Close TensorBoard writer
        writer.close()

        # Write results to file
        file = open(f'{path_net}loss_acc.txt', 'w')
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
    argparser.add_argument('--seed', type=int, help='random seed', default=64)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../../data/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    args = argparser.parse_args()

    main()
