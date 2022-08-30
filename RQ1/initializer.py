import tensorflow as tf
import numpy as np
from keras import backend as K
if __name__ == '__main__':

    dataset = 'cifar'
    model_name = 'vgg16'

    num = 50000
    epoch = 15
    seed = 42
    R = 3

    from keras.models import load_model

    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.save('new_model/' + dataset + '/'+ model_name + '/{}/init_model.h5'.format(R))