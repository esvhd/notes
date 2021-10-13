# Random Collection of DL Training Tricks...


## Kaiming Initialisation

Post [here](https://towardsdatascience.com/what-is-weight-initialization-in-neural-nets-and-why-it-matters-ec45398f99fa), original [paper](https://arxiv.org/abs/1502.01852)

Works well with deeper ReLU networks.

Pytorch has this built in. Use `fan_in` for forward pass, `fan_out` for backward pass.

```
# pytorch methods
pytorch.nn.init.kaiming_normal_()
pytorch.nn.init.kaiming_uniform_()
```

## TODO: Multi-Sample Dropout

## XLNet

[paper](https://arxiv.org/abs/1906.08237)

## Regularization is all you Need (2021)

Overview of regularisation:

- Weight decay - decouple regularisation from loss function, and applying it
after the learning rate computation (2019 paper) -
[`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

- Data Augmentation: Cut-Out, Mix-Up, Cut-Mix. Aug-Mix, Auto-Augment in RL,
  adversarial attach strategies (e.g. FGSM).

- Model Averaging: ensembled models, dropout, Mix-Out extends dropout by statistically fusing the parameters of two base models. Snapshot ensembles.

- Structural and Linearisation: ResNet's skip connections. Linearisation uses skip
  connections to transfer embeddings from previous less non-linear layers.
  Shake-Shake regularisation uses skip connections in parallel convolutional blocks.
  Shake-Drop.

- Implicit: Batch Norm, early stopping, learning rate scheduling schemes to stablise
  training convergence. Recent methods of stochastic weight averaging, or update
  in the direction of a few look-ahead steps.

The key idea of this paper, is to add $K$ regularisation methods into a single
network, then train hyperparameters $\lambda^{(k)} \forall k \in {1, cdots, K}$ to
control how much of every regularisation to use for a given problem with a
validation set.

For each hyperparameter set $\lambda^k$, there is a conditional hyperparameter
controlling whether the $k$-th regulariser is applied at all or skipped.

The paper combines 13 regularisation methods in one model. Uses BOHB for
hyperparameter optimisation.

Some regularisation cannot be combined, so apply following constraint:

1. Shake-Shake and Shake-Drop are not simultaneously active since the latter
   builds on the former
2. Only 1 data augmentation technique can be active at once due to library
   limitations. See paper reference no. 50.

Experiment setup 5.2:

- 9 layer MLP with 512 units for each layer
- learning rate `1e-3`, perform grid search to find best value across datasets.
- use AdamW optimiser, with cosine annealing with restarts as learning rate
  scheduler.
