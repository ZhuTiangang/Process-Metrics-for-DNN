# RQ3: Overfitting
Before the experiment, please be sure that the data is well prepared according to the tutorial in the main page of the repository.

## 1. Initializer
Do the same process as in RQ1 or just copy the 'new_model' folder from '/RQ1' if you have already got the initialized models in it.

## 2. Iterate
Confirm the *dataset*, *model_name* and *num* (6000 for mnist, 7326 for svhn, and 5000 for cifar) are correct, then run 'iterate.py'.
The iterative models will be stored as ('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i)), 
and the training history data will be stored as ('evaluate_' + dataset + '.xlsx').

## 3. Compute Coverage
Please make sure the values of *dataset*, *model_name*, *R* and *seed* in 'compute_coverage.py' keep the same as in 'iterate.py'. 

Then run 'compute_coverage.py', the coverage values will be stored in ('Coverage of {}.xlsx'.format(dataset)).

## 4. Correlation
Please make sure the values of *dataset*, *model_name*, *R* and *seed* in 'compute_coverage.py' keep the same as in 'compute_coverage.py'. 

Then run 'correlation.py', the Spearman and Pearson correlation results will be stored in ("correlation_" + dataset + ".txt")

## 5. Modeling
Before build a model for overfitting rates and coverage metrics, please collect all the data and put together like 'fit.xlsx'. 

Then run 'fit.py', the performance data of the model is stored in "fit.txt".
