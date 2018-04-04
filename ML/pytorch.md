# Notes for PyTorch

## Batch Norm

This [post](https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/15) discusses using batch norm at train
and test time. To stop the running average, call `eval()` function of the
batch norm layer module.


## Contiguous Tensors

A contiguous tensor is a tensor that occupies a full block of memory,
i.e. no holes in it. Views in `pytorch` can only be created on contiguous
tensors. To do so, call the `tensor.contiguous()` method.

See post [here](https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930)


## Slicing Tensors with Boolean Mask

The closest is `torch.masked_select()`, see [here](http://pytorch.org/docs/master/torch.html?highlight=masked_select#torch.masked_select).

```
x = torch.randn(10)
masks = ...
torch.masked_select(x, masks)
```
