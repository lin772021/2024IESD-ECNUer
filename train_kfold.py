import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET_kfold, ToTensor, create_dataset_kold  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import AFNet
# from sklearn.model_selection import KFold


def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []

    # Start dataset loading
    trainset = ECG_DataSET_kfold(root_dir=path_data, indice_dir=path_indices, mode='train_kfold', size=SIZE, transform=ToTensor())
    
    print("Start training")
    best_val_acc = 0.0
    for fold_index in trainset.fold_indices:
        trainloader, valloader = create_dataset_kold(trainset, BATCH_SIZE, fold_index)
        # Instantiating NN
        net = AFNet()
        optimizer = optimizers.Adam(learning_rate=LR)
        loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

        for epoch in range(EPOCH):
            running_loss = 0.0
            correct = 0.0
            accuracy = 0.0
            i = 0
            for step, (x, y) in enumerate(trainloader):
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

            running_loss = 0.0
            accuracy = 0.0

            correct = 0.0
            total = 0.0
            i = 0.0
            running_loss_val = 0.0
            for x, y in valloader:
                logits = net(x, training=False)
                val_loss = loss_object(y, logits)
                pred = tf.argmax(logits, axis=1)
                total += y.shape[0]
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                running_loss_val += val_loss
                i += x.shape[0]

            print('Val Acc: %.5f Val Loss: %.5f' % (correct / total, running_loss_val / i))

            Val_loss.append(running_loss_val / i)
            Val_acc.append((correct / total))
            if correct / total > best_val_acc:
                best_val_acc = correct / total
                # Save model
                net.save('./saved_models/kfold_5.h5')

    # Write results to file
    file = open('./saved_models/kfold_loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Val_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Val_acc))
    file.write('\n\n')

    print('Finish training')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../data/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()
