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
