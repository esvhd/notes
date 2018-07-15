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


## Tensor Swap Axes

For the equivalent of `np.swapaxes()`: `permute([0, 2, 1])`


## Slicing Tensors with Boolean Mask

The closest is `torch.masked_select()`, see [here](http://pytorch.org/docs/master/torch.html?highlight=masked_select#torch.masked_select).

```
x = torch.randn(10)
masks = ...
torch.masked_select(x, masks)
```


## 0.4 migration guide

Use `x.item()` to get the value of a 0-dim tensor.

To access underlying tensor, use `x.data` or `x.detach()`. The former is
unsafe, as any changes to `x.data` won't be tracked by 'autograd'. `x.detach()`
is the recommended way.

To move a GPU tensor to numpy:

```
gpu_tensor.detach().cpu().numpy()
```

Device agnostic code:

```
# device string format: '{device_type}:{device_ordinal}'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create tensor
x = torch.LongTensor([2, 3, 4], device=device)

# or:
x = x.to(device)

# create model
model = MyModule(...).to(device)
```

## Saving Models

Useful [post](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610)

Can save a dict of states, see example in post above. 

```
state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(state, 'model.pt')
```

Docs [recommendations](https://pytorch.org/docs/stable/notes/serialization.html#recommended-approach-for-saving-a-model)


