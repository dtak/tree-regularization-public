# NumPy Autograd Implementation

This repository includes a toy signal-and-noise HMM dataset and a basic implementation of the Tree-regularized GRU model in NumPy Autograd. Please find the ArXiv copy here: https://arxiv.org/abs/1711.06178. 

For more on NumPy Autograd, see https://github.com/HIPS/autograd.

## Setup

Create a new conda environment and activate it.

```
conda create -n interpret python=2
source activate interpret
```

Install the necessary libraries.

```
pip install -r requirements.txt
```

## Instructions

First, generate the toy dataset. This will create a directory `./data` and populate it with Pickle files.

```
python dataset.py
```

Then, you can train the tree-regularized GRU. This will create a directory `./trained_models` and dump the trained model and a PDF of the final decision tree. You can set the regularization strength as a command line argument. `apl` stands for average path length.

```
python train.py --strength 1000.0
```

We can apply the trained model on a held-out test set. This will print out AUC statistics.

```
python test.py
```

Please reach out with any concerns or potential bugs.