# RQ3: Overfitting
Before the experiment, please be sure that the data is well prepared according to the tutorial in the main page of the repository.

## 1. Initializer
Do the same process as in RQ1 or just copy the 'new_model' folder from '/RQ1' if you have already got the initialized models in it.

## 2. Iterate
Confirm the *dataset*, *model_name* and *num* (6000 for mnist, 7326 for svhn, and 5000 for cifar) are correct, then run 'iterate.py'.
The iterative models will be stored as ('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i)), 
and the training history data will be stored as ('evaluate_' + dataset + '.xlsx').

## 3. 
