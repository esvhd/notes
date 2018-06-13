# Thoughts and Notes on Papers I Read


## CNN for Sequence Modeling

Recently spending a lot of time reading about papers on sequence modeling.

The TCN [paper](https://openreview.net/forum?id=rk8wKk-R-)

Someone pointed out some related work [here](https://arxiv.org/abs/1703.04691).
Had a quick look, didn't go over the details. But the training data of 750
daily data points vs 350 validation set seemed very small to me for fitting
neural networks.

[WaveNet 2016](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
had enormous success with audio data. Would like to read about this paper next.


## Transformers

Read MIT's blog on Transformers with PyTorch.

And OpenAI's blog [here](https://blog.openai.com/language-unsupervised/).

## Language Modeling

Two papers from Stephen Merity, et al. that look very interesting and showed
SOTA results using LSTM architectures.

Notes [here](./papers.ipynb#awd_lstm)

* [Regularizing and Optimizing LSTMLanguage Models](https://openreview.net/forum?id=SyyGPP0TZ), ICLR Jan 2018
    * Model named as AvSGD Weight-Dropped LSTM (AWD-LSTM)
    * DropConnect [paper](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)
    * Need to make some notes on this paper and refer to that here.
* [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240), March 2018
    * Compare [QRNN (Bradbury et al., 2017)](https://openreview.net/forum?id=H1zJ-v5xl), LSTM with other SOTA models

Code for both models are on [github](https://github.com/salesforce/awd-lstm-lm)

Variational dropout for PyTorch doesn't seem to be readily available. Need to
look into the source code for this more.

However, Tensorflow / Keras has it already. See `recurrent_dropout` parameter
for Keras LSTM layer.

Read: https://arxiv.org/abs/1512.05287
https://discuss.pytorch.org/t/dropout-for-rnns/633　- some discusion on implementing this in pytorch.


One other paper that was cited is from [Melis, et al.](https://openreview.net/forum?id=ByJHuTgA-). On the todo list...

A few follow ups for this paper.

* Breaking the softmax bottleneck (on split cross entropy loss?)
* Neural cache pointer by Edouard Graves 2016 [link](https://arxiv.org/abs/1612.04426)
* QRNN by James Bradbury
* Melis new paper on dropout: Pushing the bounds of dropout
* Analysis of Neural Language Modeling at Multiple Scales

## Causal Impact

Read Causal Impact paper [here](https://ai.google/research/pubs/pub41854)

Concepts:

* Bayesian structural time series models
* Counterfactual estimation

Questions I have is around how to evaluate prediction
accuracy.

* How does the length of historical data affect the counterfactual estimates?


* YouTube talk by author mentioned that one way to check is to look at past
historical relationship where there wasn't any causal change, and see how the
model estimations do. But this seems a bit arbitary, is there a more
systematic way?


* Any further research on analysis of multiple causal events around the same
time?


## ICRL 2018

https://openreview.net/forum?id=HkwZSG-CZ
https://openreview.net/forum?id=SkFqf0lAZ


# Other papers notable

* [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974) - what factors lead to better feature transfer?


* PyTorch internals in a [post](https://discuss.pytorch.org/t/dynamically-expandable-networks-resizing-weights-during-training/19218)

# Tools

* Horovod by Uber - distributed training
* MLflow by Databricks - ML workflow
