import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from keras import backend as K
if __name__ == '__main__':

    dataset = 'mnist'
    model_name = 'lenet1'

    num = 5000
    epoch = 20
    seed = 46
    R = 1

    from keras.models import load_model

    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())

    # def init_layer(layer):
    #     session = K.get_session()
    #     weights_initializer = tf.variables_initializer(layer.weights)
    #     session.run(weights_initializer)
    #
    # for layer in model.layers:
    #     init_layer(layer)

    import os
    os.makedirs('new_model/' + dataset + '/' + model_name + '/{}'.format(R),exist_ok=True)

    model.save('new_model/' + dataset + '/'+ model_name + '/{}/init_model.h5'.format(R))