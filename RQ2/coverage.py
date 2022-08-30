import argparse
import os
import random
import shutil
import warnings
import sys
warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
#from PIL import Image, ImageFilter
#from skimage.measure import compare_ssim as SSIM
import keras
#from util import get_model

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"





# helper function
def get_layer_i_output(model, i, data):
    layer_model = K.function([model.layers[0].input], [model.layers[i].output])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))    #行数为num的矩阵
    return ret

# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('../data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('../data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('../data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('../data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test


class Coverage:
    def __init__(self, model, x_train,  x_adv):
        self.model = model
        self.x_train = x_train
        self.x_adv = x_adv
    
    # find scale factors and min num
    def scale(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]      #测试用例个数（矩阵第一维的长度）
        factors = dict()
        for i in layers:
            begin, end = 0, batch
            max_num, min_num = np.NINF, np.inf 
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])     #第i层的输出矩阵
                tmp = layer_output.max()
                max_num = tmp if tmp > max_num else max_num
                tmp = layer_output.min()
                min_num = tmp if tmp < min_num else min_num
                begin += batch
                end += batch
            factors[i] = (max_num - min_num, min_num)
        return factors


    #1 Neuron Coverage
    def NC(self, layers, threshold=0., batch=1024):
        factors = self.scale(layers, batch=batch)
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape   #模型第i层张量
            neuron_num += np.prod(out_shape[1:])            #累乘
        neuron_num = int(neuron_num)
        
        activate_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            buckets = np.zeros(neurons).astype('bool')
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                # scale the layer output to (0, 1)
                layer_output -= factors[i][1]
                layer_output /= factors[i][0]
                col_max = np.max(layer_output, axis=0)
                begin += batch
                end += batch
                buckets[col_max > threshold] = True
            activate_num += np.sum(buckets)
        # print('NC:\t{:.3f} activate_num:\t{} neuron_num:\t{}'.format(activate_num / neuron_num, activate_num, neuron_num))
        return activate_num / neuron_num, activate_num, neuron_num

    #2 k-multisection neuron coverage, neuron boundary coverage and strong activation neuron coverage
    def KMNC(self, layers, k=10, batch=1024):
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)
        
        covered_num = 0
        l_covered_num = 0
        u_covered_num = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            data_num = self.x_train.shape[0]
            
            neuron_max = np.full(neurons, np.NINF).astype('float')
            neuron_min = np.full(neurons, np.inf).astype('float')
            while begin < data_num:
                layer_output_train = get_layer_i_output(self.model, i, self.x_train[begin:end])
                batch_neuron_max = np.max(layer_output_train, axis=0)
                batch_neuron_min = np.min(layer_output_train, axis=0)
                neuron_max = np.maximum(batch_neuron_max, neuron_max)
                neuron_min = np.minimum(batch_neuron_min, neuron_min)
                begin += batch
                end += batch
            # print('neuron',neuron_max,'\n',type(neuron_max),neuron_max.shape,'\n',neuron_min)
            buckets = np.zeros((neurons, k + 2)).astype('bool')
            interval = (neuron_max - neuron_min) / k
            # print(interval)
            begin, end = 0, batch
            data_num = self.x_adv.shape[0]
            while begin < data_num:
                layer_output_adv = get_layer_i_output(self.model, i, self.x_adv[begin: end])
                layer_output_adv -= neuron_min
                layer_output_adv /= (interval + 10 ** (-100))

                layer_output_adv = layer_output_adv + 1
                layer_output_adv[layer_output_adv < 1.] = 0
                layer_output_adv[layer_output_adv == 11.] = 10
                layer_output_adv[layer_output_adv > 11.] = 11
                layer_output_adv = layer_output_adv.astype('int')



                # print('layer_output4', '\n', layer_output_adv[0:100], '\n', type(layer_output_adv), layer_output_adv.shape)
                for j in range(neurons):
                    uniq = np.unique(layer_output_adv[:, j])
                    # print(layer_output_adv[:, j])
                    # print(uniq)
                    buckets[j, uniq] = True
                    #print(buckets)
                begin += batch
                end += batch

            covered_num += np.sum(buckets[:,1:-1])
            u_covered_num += np.sum(buckets[:, -1])
            l_covered_num += np.sum(buckets[:, 0])
        #print('KMNC:\t{:.3f} covered_num:\t{}'.format(covered_num / (neuron_num * k), covered_num))
        #print('NBC:\t{:.3f} l_covered_num:\t{}'.format((l_covered_num + u_covered_num) / (neuron_num * 2), l_covered_num))
        #print('SNAC:\t{:.3f} u_covered_num:\t{}'.format(u_covered_num / neuron_num, u_covered_num))
        return covered_num / (neuron_num * k), (l_covered_num + u_covered_num) / (neuron_num * 2), u_covered_num / neuron_num, covered_num, l_covered_num, u_covered_num, neuron_num*k

    #3 top-k neuron coverage
    def TKNC(self,layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)
        
        pattern_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            pattern_set = set()     #集合（无重复元素）
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end]) #得到第i层的输出矩阵
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]                #每行最大k个数的下标
                topk = np.sort(topk, axis=1)                                            #将下标由小到大排列
                # or in order
                #topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                for j in range(topk.shape[0]):
                    for z in range(k):
                        pattern_set.add(topk[j][z])
                begin += batch
                end += batch
            pattern_num += len(pattern_set)
        #print('TKNC:\t{:.3f} pattern_num:\t{} neuron_num:\t{}'.format(pattern_num / neuron_num, pattern_num, neuron_num))
        return pattern_num / neuron_num, pattern_num, neuron_num

    #4 top-k neuron patterns 
    def TKNP(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]
        def to_tuple(x):        #将列表转化为元组
            l = list()
            for row in x:
                l.append(tuple(row))
            return tuple(l)

        pattern_set = set()     #集合，无重复元素
        layer_num = len(layers)
        data_num = self.x_adv.shape[0]
        patterns = np.zeros((data_num, layer_num, k))
        layer_cnt = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                #topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                patterns[begin:end, layer_cnt, :] = topk
                begin += batch
                end += batch
            layer_cnt += 1
        
        for i in range(patterns.shape[0]):
            pattern_set.add(to_tuple(patterns[i]))
        pattern_num = len(pattern_set)
        #print('TKNP:\t{:.3f}'.format(pattern_num))
        return pattern_num
    
    def all(self, layers, batch=100):
        self.NC(layers, batch=batch)
        self.KMNC(layers, batch=batch)
        self.TKNC(layers, batch=batch)
        self.TKNP(layers, batch=batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coverage of DNN')
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'adv_lenet1', 'adv_lenet4', 'adv_lenet5', 'adv_resnet20','adv_vgg16', 'svhn_model', 'adv_svhn_model', 'svhn_first', 'adv_svhn_first', 'svhn_second','adv_svhn_second'])
    parser.add_argument('-attack', help="attack model", choices=['CW', 'PGD'])
    parser.add_argument('-layer', help="test layer", type=int)

    args = parser.parse_args()
    #
    x_train, y_train, x_test, y_test = load_data(args.dataset)

    # from util import get_data
    # x_train, y_train, x_test, y_test = get_data(args.dataset)

    x_adv = np.load('./data/' + args.dataset + '_data/model/' + args.model + '_' + args.attack + '.npy')

    # ## load Xuwei's trained svhn model
    # model = get_model(args.dataset, True)
    # model.load_weights('./data/' + args.dataset + '_data/model/' + args.model + '.h5')
    #
    # model.compile(
    #     loss='categorical_crossentropy',
    #     # optimizer='adadelta',
    #     optimizer='adam',
    #     metrics=['accuracy']
    # )

    # ## load mine trained model
    from keras.models import load_model

    model = load_model('../data/' + args.dataset + '_data/model/' + args.model + '.h5')
    model.summary()

    #evaluate = AttackEvaluate(model, x_test, y_test, x_adv)
    #evaluate.robust_image_compression()
    if args.dataset == 'mnist':
        #l = [11]
        #conv
        # l = [args.layer]
        l = [0, args.layer]
    elif args.dataset == 'cifar':
        l = [1, args.layer]
    elif args.dataset == 'svhn':
        # l = [11, 15]
        l = [0, args.layer]
    coverage = Coverage(model, x_train, y_train, x_test, y_test, x_adv)
    nc1, activate_num1, total_num1 = coverage.NC(l, threshold=0.1)
    nc2, activate_num2, total_num2 = coverage.NC(l, threshold=0.3)
    nc3, activate_num3, total_num3 = coverage.NC(l, threshold=0.5)
    nc4, activate_num4, total_num4 = coverage.NC(l, threshold=0.7)
    nc5, activate_num5, total_num5 = coverage.NC(l, threshold=0.9)
    tknc, pattern_num, total_num6 = coverage.TKNC(l)
    tknp = coverage.TKNP(l)
    kmnc, nbc, snac, covered_num, l_covered_num, u_covered_num, neuron_num = coverage.KMNC(l)

    with open("coverage_result.txt", "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {} {} is: \n'.format(args.dataset, args.model, args.attack))
        f.write('NC(0.1): {}  activate_num: {}  total_num: {} \n'.format(nc1, activate_num1, total_num1))
        f.write('NC(0.3): {}  activate_num: {}  total_num: {} \n'.format(nc2, activate_num2, total_num2))
        f.write('NC(0.5): {}  activate_num: {}  total_num: {} \n'.format(nc3, activate_num3, total_num3))
        f.write('NC(0.7): {}  activate_num: {}  total_num: {} \n'.format(nc4, activate_num4, total_num4))
        f.write('NC(0.9): {}  activate_num: {}  total_num: {} \n'.format(nc5, activate_num5, total_num5))
        f.write('TKNC: {}  pattern_num: {}  total_num: {} \n'.format(tknc, pattern_num, total_num6))
        f.write('TKNP: {} \n'.format(tknp))
        f.write('KMNC: {}  covered_num: {}  total_num: {} \n'.format(kmnc, covered_num, neuron_num))
        f.write('NBC: {}  l_covered_num: {}  u_covered_num: {} \n'.format(nbc, l_covered_num, u_covered_num))
        f.write('SNAC: {} \n'.format(snac))

