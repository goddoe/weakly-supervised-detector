# Weakly Supervised Object Detector
Implementation of "Learning Deep Features for Discriminative Localization"

## Download data 

Download data with the script below.
The script download tiny-imagenet data and transform to pickle and tfrecord.
The pickle include datasets and meta information of label dictionary.
The tfrecord contains datasets; train dataset, valid dataset, test dataset.

```
python3 download_data.py
```

## Requirements

+ python3.4+
+ pip3
+ python packages - requirements.txt
+ tensorflow 1.4+


## How to train

```
$ jupyter lab

and open train.ipynb

or 

use snippet_train_model.py
```

##  How to test

```
$ jupyter lab

and open load_and_test.ipynb
```


- - -
Author
+ Sung-ju Kim
+ goddoe2@gmail.com

