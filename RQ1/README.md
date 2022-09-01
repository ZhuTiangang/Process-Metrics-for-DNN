# RQ1: Size of training datasets

Before the experiment, please be sure that the data is well prepared according to the tutorial in the main page of the repository.

## 1. Initializer
Vary the 'dataset' and 'model_name' in the main body of 'initializer.py' to chose a experimented model, for example:

```
dataset = mnist
model_name = lenet1
```

Then, run 'initializer.py', the initialized model will be stored as ('new_model/' + dataset + '/'+ model_name + '/{}/init_model.h5'.format(R)). 
The parameter R is to store different initialized models for repeating experiments, which is varied from 1 to 3 (or more).

## 2. Iterate
Please make sure the values of 'dataset', 'model_name' and R in 'iterate.py' keep the same as in 'initializer.py', and fill in the value of 'num' corresponding to the size of dataset (60000 for mnist, 73257 for svhn, and 50000 for cifar). 

Then, run 'iterate.py', the iterative models will be stored as ('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i)). The parameter seed is to shuffle the dataset for repeating experiments.

## 3. Coverage trend
Please make sure the values of 'dataset', 'model_name', R and seed in 'coverage_trend.py' keep the same as in 'iterate.py'. 

Then run 'coverage_trend.py', the coverage trends will be stored in ('coverage_result_of_train_data.txt').

For insufficient dataset, please change the value of 'num' (6000 for mnist, 7326 for svhn, and 5000 for cifar), and rerun the step 2, 3.
