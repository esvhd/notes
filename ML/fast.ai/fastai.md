# fast.ai Notes

Some notes based on Practical Deep Learning for Coders 2019 v3.

## ML Tips

Use random seeds to create a fixed validation set which is then used in parameter turning or other testing. This makes validation loss more comparable. Otherwise loss would be computed over different validation data. 

### One-Cycle Learning Rate

Increase learning rate gradually in the beginning, then reduce it over epoches. What we define as learning rate here is really the **maximum** learning rate in this cycle.

**Before** unfreezing the last few layers, use `lr_find()` and pick a learning rate where the curve is **decling fastest**.

**After** unfreezing the last few layers, run `lr_find()`, choose the point when curve starts to pick up, then **divide by 10**, use this as the lower bound. Use previous LR (frozen) / 5 as upper slice.

### Discriminative Learning Rate

Train different layers with different learning rates. E.g.

```
learn.fit_one_cycle(5, slice(1e-2/(2.6**4), lr/5))
```

Note that the slice would specify the **max** learning rate for the first and last layer. 

`lr/2.6**4`: 2.6 came from Jeremy Howard's research work, he trained a large variety of models with different hyperparameters, then used a random forest to learn what parameters would give better prediction results. 

### RNN

**Lower** momentum helps with RNN training, IMDB model uses `moms=(0.8, 0.7)`. In one-cycle training schedule, momentum goes from `moms[0]` to `moms[1]` in phase 1, then back to moms[0] in phase 2.

### Image Segmentation

Use `u-net`


## Tabular Problems with NN

TODO: Lesson 4


## API Tips

To get help for an API function use `help(func)` to see docstrings.

By default `fastai` uses GPUs, to change to cpu for inference, use:

```
fastai.defaults.device = torch.device('cpu')
```

`doc(func)` to show documentation.

`untar_data`: downloads and untar data.

`path.ls()` - `ls` for this path. super useful.

Uses `pathlib`

`ImageDataBunch`: object to represent what you need for training. 
    * Set `np.random.seed()` before using `ImageDataBunch` to create validation set.
    * `.from_name_re()` extracts labels from file names using regex. 
    * Data normalization can be done by calling `data.normalize()`, where `data` is returned by `.from_name_re()` above.
    * `data.c` for classification problems is the number of classes.
    * Image size, generall `size=224` works... Mentioned in lession one ~0:29:00. 
    * batch size for memory issues, `bs=`


`Learner` used to instantiate models. 
    * can use pre-trained models
    * `.fit_one_cycle(cycle_length=)` is prefored to `fit()` since 2018.
    * `.save()` to save model, files are saved under `~/.fastai/models`
    * `.lr_find()`
    * `to_fp16()` for floating point 16 training 
    * `freeze_to()` to freeze to a certain layer


`learner.recorder` methods:
    * `plot()` LR finder loss vs learning rate
    * `plot_lr()` plots learning rate
    * `plot_losses()` plots training and validation losses


`ClassificationInterpretation.from_learner(learner)`:
    * `plot_top_losses()`
    * `most_confused(min_val=2)`


### Basic Block for CNN

```
# TODO
from fastai import *

learn = create_cnn(data, models.resnet34)
learn.loss_func = MSELossFlat()
learn.lr_find()
learn.recorder.plot()

# pick lr based on plot
lr = 1e-4
learn.fit_one_cycle(5, slice(lr))

learn.save('model-1')
# show predicted resuts
learn.show_results()
```


## Other Tips

`starlette` is a useful tool to create light weight webapps, like `flask`. Also [Responder](#https://github.com/kennethreitz/responder) is a tool created on top of `starlette`.


## Lession 3

Image segmentation, image regression problems, text classification.
