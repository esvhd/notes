# fast.ai Notes

Some notes based on Practical Deep Learning for Coders 2019 v3.

<!-- MarkdownTOC levels="1,2,3,4" autolink=true -->

- [ML Tips](#ml-tips)
    - [Encode `NA` as Binary Flags](#encode-na-as-binary-flags)
    - [One-Cycle Learning Rate](#one-cycle-learning-rate)
    - [Discriminative Learning Rate](#discriminative-learning-rate)
        - [Code Details](#code-details)
    - [Image Augmentation](#image-augmentation)
    - [Datetime](#datetime)
    - [RNN](#rnn)
    - [Image Segmentation](#image-segmentation)
- [Tabular Problems with NN](#tabular-problems-with-nn)
- [API Tips](#api-tips)
    - [Preprocessing](#preprocessing)
    - [General](#general)
    - [`DataBunch`](#databunch)
    - [`Learner`](#learner)
    - [Regularization](#regularization)
        - [Dropout](#dropout)
    - [Basic Block for CNN](#basic-block-for-cnn)
- [`pytorch`](#pytorch)
- [Other Tips](#other-tips)
- [Lession 3](#lession-3)

<!-- /MarkdownTOC -->


## ML Tips

Use random seeds to create a fixed validation set which is then used in parameter turning or other testing. This makes validation loss more comparable. Otherwise loss would be computed over different validation data. 

### Encode `NA` as Binary Flags

Use a separate binary feature to represent NA / missing data.

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

#### Code Details

When applying discriminative learning rates, instead of using a different learning rate for each layer, `fastai` divides a network into groups. Each grop would use the same learning rate.

Supported ways to specifiy learning rates are:

* single numerical value, e.g. 1e-3, same learning rate everywhere.
* `slice(1e-3)`, final layer group gets `1e-3`, other layers use `1e-3 / 3`.
* `slice(1e-5, 1e-3)`, first layer groups uses `1e-5`, final layer group uses `1e-3`. Layers inbetween uses multiplicatively equal values between those values, e.g. for 3 groups, the middle group uses `1e-4`.

For CNNs, typically we have 3 layer groups. The last layer group consists of the head, the rest of the network is divided into 2 groups. 

### Image Augmentation

Border reflection is very useful. This creates mirror images of the image borders. (Lesson 6)

Imaging wrapping also very useful. 


### Datetime

Always encode additional features for datetime columns. This allows a model to capture seasonalities for example.

```
# from fastai dl1 course rossmann notebook
def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
```

### RNN

**Lower** momentum helps with RNN training, IMDB model uses `moms=(0.8, 0.7)`. In one-cycle training schedule, momentum goes from `moms[0]` to `moms[1]` in phase 1, then back to moms[0] in phase 2.

### Image Segmentation

Use `u-net` - it has a smiliar idea as ResNet / dense net, for the down-sampling and up-sampling processes that had the same dimension, a skip connection is added to the up-sampling output. However, instead of adding, concat was used.

**De-convolution** - increase the output size, the reverse of convolution. E.g. using nearest neighbour interpolation (lesson 7).


## Tabular Problems with NN

TODO: Lesson 4, 5, 6 (Rossmann)

General architecture of tabulr learner's model is to use embedding layers for categorical features, and `BatchNorm1d` for continuous features, concatenate embeddings with the output of `BatchNorm1d`. 

Default embedding dim `num_cat, min(600, round(1.6 * num_cat**0.56))`. A bit of magic... Each categorical variable gets its own embedding layer.


## API Tips

### Preprocessing

`Categorify()`: `pandas` uses -1 to encode `NaN` or missing category, however, this won't directly work for embedding lookup. `fastai` uses `pandas` categorical values + 1, i.e., 0 represents `NaN`.


### General

To get help for an API function use `help(func)` to see docstrings.

By default `fastai` uses GPUs, to change to cpu for inference, use:

```
fastai.defaults.device = torch.device('cpu')
```

`defaults.cmap=` defines the default colour map.

`doc(func)` to show documentation.

`untar_data`: downloads and untar data.

`path.ls()` - `ls` for this path. super useful.

Uses `pathlib`

`fastai` adds a `.pca()` method to `torch.Tensor`! 

### `DataBunch`

Some coding conventions:

* `train_ds` - training `TensorDataset`
* `data = DataBunch.create(train_ds, valid_ds, bs=batch_size)` creates data loaders. E.g. `data.train_dl` is the training data dataloader.


`ImageDataBunch`: object to represent what you need for training. 

* Set `np.random.seed()` before using `ImageDataBunch` to create validation set.
* `.from_name_re()` extracts labels from file names using regex. 
* Data normalization can be done by calling `data.normalize()`, where `data` is returned by `.from_name_re()` above.
* `data.c` for classification problems is the number of classes.
* Image size, generall `size=224` works... Mentioned in lession one ~0:29:00. 
* batch size for memory issues, `bs=`


```
# lesson 6, Rossmann problem

TabularList.from_df(...)
           .split_by_idx(...)
           # log=True log transform of the dependent variables / y's
           .label_from_df(cols=dep_vars, label_cls=FlatList, log=True)
           .databunch()
```

Taking `log` of dependent variables / `y`, essentially converts a RMSPE (root mean square percent error) to RMSE. Therefore, the network can be trained with normal RMSE loss.

`.label_from_folder()`: uses folder names as labels, e.g. MNIST dataset. Returns `LabelLists`.

`LabelLists` has a `.transform()` funciton, which takes a list of 2 lists of transformations, one for training set, one for validation set. (Lesson 7 MNIST notebook)

TODO:

* work out ItemList methods
* Learner contains data, model, and optimizer, everything needed for training.

### `Learner`

`Learner` used to instantiate models. 

* Useful way to construct a network: `Learner(data, MNIST_NN(), loss_func=loss_func, metrics=accuracy)`, then the useful functions all becomes available.
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


`tabular_learner()` parameters:

* `ps=` fully connected dropout probabilities
* `emb_drop` embedding dropout rate
* `y_range=` should use ranges slightly smaller / larger than the range of `y` TODO: check lesson 5 Rossmann section
* embedding size dict to specify embedding dimension. `fastai` defaults are usually good.

**Trick** on `y_range`: `y` here is the input to a sigmoid activation. Sigmoid asymptotes at the maximum / minimum, which means if the max value you are trying to predict is say 5, then we need to adjust the Y range to be slightly outside the min / max range, so that the activation can high those values.


### Regularization

**Weight Decay**: Every learner has a `wd=` parameter. Jeremy mentioned that `0.1` was usually a good choice for `L2` weight decay. However, library defaults to `0.01` to prepare for when there is too much regularization. Always try `0.1`. Good to distinguish weight decay and `L2` or `L1` here, the penalty term here is `wd * sum(parameters**2)`, `wd` is the coefficient that controls the level of regularization.

#### Dropout

At training time, activation is dropped / zeros with a probability of `p`.

At test time, no dropout is applied. Therefore, the activations would be larger. Original paper suggests mutiplying the weight matrix by `p` to adjust.

In pytorch, dropout masks are modified at training time, resulting in larger activation values and hence smaller weights (in order to get to the same training targets), so at test time no additional adjustment is needed. Implementation code:

```{C}
// bernoulli binary mask of 0s and 1s
noise.bernoulli_(1 - p);

// divided by 1 - p to scale the mask
noise.div_(1 - p);

return multiply<inplace>(input, noise);
```


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


`conv_layer()`: shortcut to create a block of convoluation, batch norm and relu layers.

`res_block`: returns residual block `out = x + conv2(conv1(x))`

`MergeLayer`: can handle either residual block (add `x`) or dense block (concatenates `x`). DenseNet works very well for **smaller**
datasets.

## `pytorch`

- `Flatten()` layer removes unit axis, e.g. from input dim `(10, 1, 1)` to `(10,)`.

TODO: hooks

## Other Tips

`starlette` is a useful tool to create light weight webapps, like `flask`. Also [Responder](#https://github.com/kennethreitz/responder) is a tool created on top of `starlette`.

`pandas.read_csv(..., encoding='latin-1')` - This is helpful in decoding files not using unicode. 


## Lession 3

Image segmentation, image regression problems, text classification.
