# PFA-learner
This is the sample code for our SIGMORPHON 2021 paper "Simple induction of (deterministic) probabilistic finite-state automata for phonotactics by stochastic gradient descent" (Dai & Futrell, to appear)

Prerequisites: Python 3, PyTorch.

When `pfa.py` is run on the command line, it will fit and evaluate a PFA on the desired training and testing data. Training results are output to `stdout`.

Usage:

```
usage: pfa.py [-h] [--lang LANG] [--model_class MODEL_CLASS] [--num_epochs NUM_EPOCHS]
              [--nondeterminism_penalty NONDETERMINISM_PENALTY] [--memory_mi_penalty MEMORY_MI_PENALTY]
              [--batch_size BATCH_SIZE] [--init_temperature INIT_TEMPERATURE] [--lr LR] [--activation ACTIVATION]
              [--num_samples NUM_SAMPLES] [--perm_test_num_samples PERM_TEST_NUM_SAMPLES] [--print_every PRINT_EVERY]
              [--seed SEED]

Induce and evaluate a PFA to model forms.

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           language data to use (by default Quechua)
  --model_class MODEL_CLASS
                        model class (integer, sp, sl, sp_sl)
  --num_epochs NUM_EPOCHS
  --nondeterminism_penalty NONDETERMINISM_PENALTY
  --memory_mi_penalty MEMORY_MI_PENALTY
  --batch_size BATCH_SIZE
                        batch size; 0 means full gradient descent with no batches
  --init_temperature INIT_TEMPERATURE
                        initialization temperature
  --lr LR               starting learning rate for Adam
  --activation ACTIVATION
                        activation function for probabilities: softmax, sparsemax, or entmax15
  --num_samples NUM_SAMPLES
                        number of samples to output
  --perm_test_num_samples PERM_TEST_NUM_SAMPLES
                        number of samples in permutation test
  --print_every PRINT_EVERY
                        print results per x epochs
  --seed SEED           random seed for train-dev-test split and batches
```
