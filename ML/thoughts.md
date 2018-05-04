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

One other paper that was cited is from [Melis, et al.](https://openreview.net/forum?id=ByJHuTgA-). On the todo list...

## ICRL 2018

https://openreview.net/forum?id=HkwZSG-CZ
https://openreview.net/forum?id=SkFqf0lAZ
