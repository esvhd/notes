# Notes for PyTorch

## Batch Norm

This [post](https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/15) discusses using batch norm at train
and test time. To stop the running average, call `eval()` function of the
batch norm layer module.
