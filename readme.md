# InterNeural: An Efficient Framework forHigh Dimensional Non-linear Contextual Bandits

## Table of Contents

- [Background](#Background)
- [Install](#Install)
- [Usage](#usage)

## Background
This project is the code implementation for the paper ***Interconnected Neural-Linear Contextual Bandits with UCB Exploration***.

## Install
This project uses [Python](https://www.python.org/downloads/release/python-383/) 3.8.3, [PyTorch](https://pytorch.org) 1.5.0, [sklearn](https://scikit-learn.org/stable/) 0.0 and [NumPy](https://numpy.org)  1.18.1.

```sh
$ pip3 install torch==1.5.0
```
```sh
$ pip3 install sklearn
```
```sh
$ pip3 install numpy==1.18.1
```

## Usage
* Navigate to the project workspace 
* Run the following command
### Experiments on Synthetic Datasets
Find the runner function for each bandit algorithm in ```InterNeuralBandit/runner.py```.
Then change hyper-parameters for each bandit algorithm.

Then run the function ```synthetic_data_experiment``` in ```InterNeuralBandit/runner.py```.

Find files for results (cumulative regret and runtime) in ```InterNeuralBandit/regret_stats/```. Files are stored in csv format. 

### Experiments on Real-World Datasets
Download three datasets [MUSIC](http://archive.ics.uci.edu/ml/datasets/FMA%3A+A+Dataset+For+Music+Analysis), [FONT](http://archive.ics.uci.edu/ml/datasets/Character+Font+Images) and [MNIST](http://yann.lecun.com/exdb/mnist/).

Pre-process each dataset to transfer it to a csv file, where each row is in the format of 'class index, features'. Each class should have the same number of instances.

Run the function ```real_data_experiment``` in ```InterNeuralBandit/runner.py``` with specified number of classes, dimension of features, dimension of latent features and number of instances per class.

Find files for results (cumulative regret and runtime) in ```InterNeuralBandit/regret_stats/```. Files are stored in csv format. 


### Efficiency test for NeuralUCB

Run the function ```efficiency_test_for_NeuralUCB``` in ```InterNeuralBandit/runner.py```.

Find files for results (cumulative regret and runtime) in ```InterNeuralBandit/regret_stats/```. Files are stored in csv format. 


