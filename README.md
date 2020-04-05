# Work in progress

## Supervised-learning-of-Many-Body-Localization
A simple neural network structure to classify many-body localized and thermalized phases using entanglement spectrum as the input data. This project is inspired by a great machine learning [review paper](https://arxiv.org/abs/1803.08823) for physics by Mehta et. al and the accompanying jupyter notebooks.

The network can take either entanglement spectrum or the wavefunctions (squared) as the 
input. Accordingly the structure of the network (# of layers, neurons etc.) will change accordingly 
to achieve the best performance. The optimal hyper-parameters can be estimated using Bayesian optimization or Hyperband method provided in [Keras Tuner](https://keras-team.github.io/keras-tuner/).

With moderate amount of data (O(10^4) entries), the classifier achieves nearly 100% accuracy on the test set.

The example Hamiltonian comes from this [paper](https://link.aps.org/doi/10.1103/PhysRevB.100.235144),
where the system is in the MBL phase if the interaction is weak (J/t<<1) and in ETH phase otherwise.

---
The strategy for establishing the phase diagram as a function of interaction strength j is:
* first train the model (a binary classifier) using the data deep in the two phases, say j=0.01 and 5.0 respectively.
* evaluate the model using different j to see its accuracy
* the phase boundary happens where the classifier fails (around %50 accuracy)

## Requirement
Python 3.5+

NumPy

Tensorflow (using Keras API) and scikit-learn

[Keras Tuner](https://keras-team.github.io/keras-tuner/) for hyper-parameter tuning
[Quspin](https://github.com/weinbe58/QuSpin) (Powerful numerical exact-diagonalization package for 1D systems)
