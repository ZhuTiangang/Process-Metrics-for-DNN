import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':


    df = pd.read_excel("fit.xlsx", sheet_name="Sheet1")
    print(df)
    df1 = pd.DataFrame(df)
    x = df1[['NC0.1', 'NC0.3', 'NC0.5', 'NC0.7', 'NC0.9', 'TKNC', 'TKNP', 'KMNC', 'NBC', 'SNAC']]
    y = df1[['y']]
    # print(type(y))
    # print(x, y)
    max = x.max(axis=0)
    min = x.min(axis=0)
    x = (x - min) / (max - min)
    max = y.max(axis=0)
    min = y.min(axis=0)
    y = (y - min) / (max - min)
    # print(x, y)
    x, y = shuffle(x, y, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    '''model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, input_shape=(10,), activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    #model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
'''
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10)
    rfr.fit(x_train, y_train)

    score = rfr.score(x_test, y_test)
    print('score:', score)
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    # print('test', y_test[:10])
    y_pred = rfr.predict(x_test)
    # print('pred', y_pred[:10])
    err = abs(y_pred - y_test)
    # print('error', err[:10])
    mape = 100 * err / (y_test + 10 ** (-100))
    # print(mape[:10])
    count = 0
    for i in mape:
        if i <= 25:
            count = count + 1

    print('mape:', np.mean(mape))
    print('less than 25% mape:', count / mape.size * 100)
    # print(err.shape)
    with open("fit.txt", "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('score: {}\n'.format(score))
        f.write('mape: {}\n'.format(np.mean(mape)))
        f.write('less than 25% mape: {}\n'.format(count / mape.size * 100))
