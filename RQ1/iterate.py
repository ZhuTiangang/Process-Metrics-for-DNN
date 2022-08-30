import tensorflow as tf
import numpy as np
from keras import backend as K
import time
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN','FASHION_MNIST'])
    name = name.lower()
    x_train = np.load('../data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('../data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('../data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('../data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test

def retrain(model, X_train, Y_train, X_test, Y_test, batch_size=128, epochs=10):

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # training without data augmentation
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
    return model, history


if __name__ == '__main__':

    dataset = 'cifar'
    model_name = 'vgg16'

    num = 50000
    epoch = 15
    seed = 6
    R = 3

    x_train, y_train, x_test, y_test = load_data(dataset)
    data_shape = x_train.shape
    print(data_shape)
    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train, random_state = seed)

    # load mine trained model
    from keras.models import load_model

    model = load_model('new_model/' + dataset + '/'+ model_name + '/{}/init_model.h5'.format(R))
    model.summary()
    #tf.compat.v1.disable_eager_execution()
    #sess = tf.compat.v1.InteractiveSession()
    #sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(epoch):
        print(i)
        print(time.asctime())
        if i >= 1:
            model = load_model('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i-1))
        model_i, history_i = retrain(model, x_train[0 : num], y_train[0 : num], x_test, y_test, batch_size=128, epochs=1)
        model = model_i
        model_i.save('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i))

        # model = load_model('new_model/' + dataset + '/'+ model_name +'/model_{}.h5'.format(i))

        with open("evaluate_result.txt", "a") as f:
            f.write("\n------------------------------------------------------------------------------\n")
            f.write('the result of {} {} epoch{} is: \n'.format(dataset, model_name,i))
            f.write('{} \n'.format(history_i.history))
        K.clear_session()