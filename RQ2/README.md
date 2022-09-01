# RQ2: Adversarial attacks
Before the experiment, please be sure that the data is well prepared according to the tutorial in the main page of the repository.
## 1. Craft adv_examples
Use the following commands to craft adv_examples for mnist lenet1 (please vary parameters *-d* and *-m* for other models):
```
$ cd RQ2
$ python craft_adv_examples.py -d mnist -m lenet1 -a all
```
The adv_examples will be stored as ('/data/Adv_%s_%s_%s.npy' % (dataset, args.model, attack))

### Attention: 
there may occur some unexpected problems with cw attack (most of the pictures show dark), if so please try following command to craft cw examples:
```
$ python attack_1.py -dataset mnist -model leent1 -attack CW
```

## 2. Initializer
Do the same process as in RQ1 or just copy the 'new_model' folder from '/RQ1' if you have already got the initialized models in it.

## 3. Adv_iterate
Confirm the *dataset*, *model_name*, *attack* and *num* are correct, then run 'adv_iterate.py'.

The iterative models will be stored as ('new_model/' + dataset + '/'+ model_name +'/{}/model_{}.h5'.format(R, i)). 
The parameter *adv_epoch* is to determine which epoch to insert adv_examples, you can modify it if you like.

## 4. Adv_coverage
Please make sure the values of *dataset*, *model_name*, *attack*, *num* and *adv_epoch* in 'adv_coverage.py' keep the same as in 'adv_iterate.py'. 

Then run 'adv_coverage.py', the coverage values will be stored in ('Coverage trend with {0} adv_examples in epoch {1}.xlsx'.format(attack, adv_epoch)).

