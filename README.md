# Supervised-learning-of-Many-Body-Localization
A simple neural network structure to classify many-body localized and thermalized phases using entanglement spectrum as the input data

With moderate amount of data O(10^4), the classifier achieves nearly 100% accuracy on the test set.

The example Hamiltonian is taken from this [paper](https://arxiv.org/abs/1802.10029),
where the system is in MBL phase when the interaction is weak (J/t<<1) and in ETH phase otherwise.

# Requirement
Python 3.5+

Tensorflow

[Quspin](https://github.com/weinbe58/QuSpin) (for calculating the entanglement spectrum)