# -*- coding: utf-8 -*-
"""
"""

from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py as h5py
from input_data import get_mnist_data_iters
# 加载字体库作为训练样本
from keras.datasets import mnist
K.set_image_data_format('channels_last')
from keras import regularizers

# 使用迁移学习的思想，以VGG16作为模板搭建模型，训练识别手写字体
# 引入VGG16模块
def VGGNet(input_shape, n_class):
    """
    A VGG Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    # 建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层（全连接层）去掉，只保留其余的网络
    # 结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
    # VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
    # VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器
    model_vgg = VGG16(include_top=False, weights=None, input_shape=input_shape)
    for layer in model_vgg.layers:
        layer.trainable = True #允许调整之前的卷积层的参数

    x = Flatten(name='flatten')(model_vgg.output)
    #x = Dense(4096, activation='relu', name='fc6')(x)
    #x = Dense(4096, activation='relu', name='fc7')(x)
    #x = Dropout(0.5, name='dropout')(x)
    y = Dense(n_class, activation='softmax', name='predictions')(x)
    #y = Dense(n_class, activation='softmax', name='predictions', kernel_regularizer=regularizers.l2(0.01))(x)

    train_model = Model(inputs=model_vgg.input, outputs=y, name='vgg16')
    eval_model = Model(inputs=model_vgg.input, outputs=y)
    return train_model, eval_model



def train(model, data, args):
    """
    :param model: the VGG-MNIST model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
                             #  write_grads=True, write_images=True)
    earlystopper = callbacks.EarlyStopping(patience=5, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))
    # 在训练和测试之前需要先编译模型
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=args.epochs, batch_size=args.batch_size, callbacks=[log, tb, earlystopper, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/vgg_MNIST_trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    return model


def vgg16_test(model, data, model_path):
    x_test, y_test = data
    _ = model.load_weights(model_path)
    y_pred = model.predict(x_test, batch_size=64)
    print(y_test.shape[0])
    accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    print('-'*50)
    print('Test accuracy:', accuracy)


# 获取训练集以及测试集，这里会从亚马逊网站上下载，国内被墙掉了，只能换用本地加载方式
def load_mnist_corrupted(data_dir='/home/yeler082/datasets/MNIST/',
                                               train_size=10000, test_size=2000, full_test_set=False, seed=20):
    # the data, shuffled and split between train and test sets
    train_set, test_set = get_mnist_data_iters(data_dir,
                                               train_size, test_size, full_test_set, seed)
    x_test, y_test = [x[0] for x in test_set], [x[1] for x in test_set]
    x_train, y_train = [x[0] for x in train_set], [x[1] for x in train_set]

    y_test = list(map(int, y_test))
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_train = list(map(int, y_train))
    x_train, y_train = np.array(x_train), np.array(y_train)

    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    # 因为VGG16对网络输入层需要接受3通道的数据的要求，我们用OpenCV把图像从32*32变成224*224，把黑白图像转成RGB图像
    # 并把训练数据转化成张量形式，供keras输入
    # convert image 28*28 --> 224*224 and i channel --> 3 channels
    img_size = 224
    x_train = [cv2.cvtColor(cv2.resize(i, (img_size, img_size)), cv2.COLOR_GRAY2RGB) for i in x_train]
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
    x_test = [cv2.cvtColor(cv2.resize(i, (img_size, img_size)), cv2.COLOR_GRAY2RGB) for i in x_test]
    x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

    x_train = x_train / 255.
    x_test = x_test / 255.
    print("*******", x_train.shape, x_test.shape)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    from keras.models import load_model
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=True, type=bool)  # True is used to train,False is used to test.
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--data_dir', default='/home/yeler082/datasets/MNIST/', type=str)
    parser.add_argument('--train_size', default=10000, type=int)
    parser.add_argument('--test_size', default=2000, type=int)
    parser.add_argument('--full_test_set', default=False, type=bool)
    parser.add_argument('--model_path', default='./result/weights_07.h5', type=str)
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--print', default=None)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist_corrupted(args.data_dir, args.train_size, args.test_size,
                                                                args.full_test_set, args.seed)

    # define model
    model, eval_model = VGGNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))))
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        vgg16_test(model=eval_model, data=(x_test, y_test), model_path=args.model_path)
