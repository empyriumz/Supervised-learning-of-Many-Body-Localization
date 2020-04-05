# Work in progress

# Supervised-learning-of-Many-Body-Localization
A simple neural network structure to classify many-body localized and thermalized phases using entanglement spectrum as the input data. This project is inspired by a great machine learning [review paper](https://arxiv.org/abs/1803.08823) for physcis by Mehta et. al and the accompanying jupyter notebooks.

With moderate amount of data (O(10^4) entries), the classifier achieves nearly 100% accuracy on the test set.

The example Hamiltonian is taken from this [paper](https://link.aps.org/doi/10.1103/PhysRevB.100.235144),
where the system is in MBL phase when the interaction is weak (J/t<<1) and in ETH phase otherwise.

# Requirement
Python 3.5+

NumPy

Tensorflow (using Keras API) and scikit-learn

[Quspin](https://github.com/weinbe58/QuSpin) (Powerful numerical exact-diagonalization package for 1D systems)
