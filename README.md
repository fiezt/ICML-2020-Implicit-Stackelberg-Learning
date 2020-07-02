## Paper Information
Implicit Learning Dynamics in Stackelberg Games:
Equilibria Characterization, Convergence Analysis, and Empirical Study, ICML 2020. Tanner Fiez, Benjamin Chasnov, Lillian Ratliff.

## Stackelberg learning dynamics

Hierarchical learning for machine learning:
```
# Leader:
min_x { f1(x,y) | y ∈ argmin_y f2(x,y) }

# Follower:
min_y { f2(x,y) }
```
where `x` and `y` are parameters associated with player 1 and 2 respectively.

Gradient-based learning:
```
# Leader:
x(t+1) = x(t) - lr1(t) * (D1f1(x(t), y(t)) - D21f2^T(x(t), y(t))(D22f2)^{-1}(x(t), y(t))D2f1(x(t), y(t))

# Follower:
y(t+1) = y(t) - lr2(t) * D2f2(x(t),y(t))
```
where the leader's update is the total derivative of cost `f1` where from the implicit function theorem we have `Dr = D22f2^-1 * D21f2`, where `D22f2^-1`.

## Installation
We recommend running
```
conda env create -f environment.yaml
```
to install a virtual environment with the correct packages.

## Experiments
The `experiments` directory includes the following experiments:
* Covariance estimation with a linear generator and a quadratic discriminator
* Bimatrix games with softmax policies
* Polynomial continuous game with zero-sum objectives


The GAN-Code directory includes all code for GAN experiments. The shell scripts can be run for experiments on
* Mixture of guassian generative adversarial networks (GANs)
* MNIST dataset generative adversarial networks.

The files contained are briefly as follows:
* ComputationalTools.py: Tools for computing JacobianVector products.
* DCGAN.py: DCGAN architecture for MNIST experiments.
* EigenvalueTools.py: Tools for computing eigenvalues of relevant game objects.
* EvaluationMNIST.ipynb: Notebook to compute inception score from checkpoints of MNIST experiments and sample fake images.
* EvaluationMoG.ipynb: Notebook to compute eigenvalues and generate figures from checkpoints of MoG experiments.
* GameGradients.py: Tools to build game Jacbian and get player gradients.
* GameLosses.py: Networks for MoG experiments helper function to generate generator and discriminator loss.
* LeaderUpdateTools.py: Functionality to compute the leader gradient update.
* LoggingTools.py: Tools for logging some statistics while running experiments.
* MNIST_Simulation.py: Main script to run MNIST experiments.
* MoG.py: Data generation for Gaussian experiments and plotting tools.
* MoG_Simulation.py: Main script to run MoG experiments.
* ObjectiveTools.py: Gan loss functions.
* Utils.py: Various simple helper functions.
* inception_evaluation.py: Code for computing inception score on MNIST.
* run_mnist.sh: Shell script to run batch of MNIST experiments.
* run_mog_batch.sh: Shell script to run batch of MoG experiments.
* train_mnist_classifier.py: Code to train MNIST classifier for inception score computation.

