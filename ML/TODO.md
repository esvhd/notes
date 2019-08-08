# TODO list

## Finance

1. Journal of Financial Data Science
2. Downloaded portfolio optmization papers in Documents folder. 3x
3. Kahn book
4. Fabozzi robust portfolio optimization book

## AWD LSTM

Notebook: `awd-lstm-ts`

Models in `git/mlstack`. training code for fx in `git/pytorch_play/finpack`.

train a model and plot `y_true` vs `y_pred` in a scatter plot or `sns.regplot`. This should show if prediction
actually reflected `y_true`, or is it just copying.

Old fx prediction scrips probably need some love to upgrade to pytorch 0.4

Another idea I have is to use awd-lstm to feed into a mixed density network,
use the output of MDN for directional trading. Concern is the the prediction error
would be propogate through the network and result in larger errors.

2018-07-12
working on `fx_awd_lstm.py`, need to finish `rnn_train()`. find out more about `repackage_hidden()`.

### Dynamic Evaluation

still need to get the code working. Filed issue on github.

## Data labels

Need to write some code to label financial data.

## ML TODO:

Noam Brown Taxes hold them paper

ML Courses [list](https://twitter.com/chipro/status/1157772112876060672)

[Efficient Net checkpoints](https://twitter.com/quocleix/status/1156334264322940928)

[R-Transformer: Recurrent Neural Network Enhanced Transformer](https://arxiv.org/abs/1907.05572), [code](https://github.com/DSE-MSU/R-transformer)

Review [TCN](https://arxiv.org/abs/1803.01271), and [Trellis network](https://arxiv.org/abs/1810.06682)

[DeepGLO](https://arxiv.org/abs/1905.03806) sequence modeling / Time Series forecasting

[Stacked Capsule Autoencoders](https://akosiorek.github.io/ml/2019/06/23/stacked_capsule_autoencoders.html)

[Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)

[Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242)

sklearn IterativeImpute

Trees: Unbiased recursive partitioning: https://eeecon.uibk.ac.at/~zeileis/news/power_partitioning/

Recurrent Transformer: https://arxiv.org/abs/1907.05572

Visualizing Recurrent Neural Networks: https://arxiv.org/abs/1506.02078

[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838)


ML financial asset return forecast `ml_planning` notebook

Class imbalance problem [here](https://t.co/Gf17gEjyqf), read this from a tweet reply to Ian Goodfellow.

ULM-fit, BERT model. Transformer XL. Open AI GPT-2. Hugging Face github repo for pre-trained BERT.

Hyperparameter Importances [paper](https://arxiv.org/abs/1710.04725), Aug 2018

Multi-Horizon Quantile RNN from Amazon [paper](https://arxiv.org/abs/1711.11053)

Attention [distill.pub](https://distill.pub/2016/augmented-rnns/)

Smerity blog on breaking the bottleneck of softmax

Fixing Adam Weight Decay Regularizaition [here](https://arxiv.org/abs/1711.05101)

Interesting paper on embedding stability [here](https://arxiv.org/abs/1804.09692)

PyTorch Tutorials on RNN
https://github.com/ritchieng/the-incredible-pytorch

echan's notes on LSTM:
http://blog.echen.me/2017/05/30/exploring-lstms/

Yoshua Bengio on RNN:
http://videolectures.net/deeplearning2016_bengio_neural_networks/

MILA Summer School videos from 2016
http://videolectures.net/deeplearning2016_montreal/

Andrea Karpathy's blog on RNN
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
https://github.com/karpathy/char-rnn

RNN material
https://github.com/kjw0612/awesome-rnn

Finish Nando de Freitas's course at UBC/Oxford

Stanford CS231
Berkeley Deep Reinforcement Learning

Fast.ai computational linear algebra course
http://www.fast.ai/2017/07/17/num-lin-alg/

Good List of Courses:
https://deeplearning4j.org/deeplearningpapers.html

Networks:
Residual Networks - linked to initalization with identity matrix
Hinton showed that ReLU is approximately a stack of logistic units.

Summarize Chan's book


### Other TODO

Move rebal_freq to backtest.backtest(). Rebal freq should be a general case.

find methods to test risk.calc_risk_stats()

test credit strategy with mixed set of signals. binary + smoothed.

2nd Smoothed signal:
transform distance with np.log(dist.abs() + 1e-8), then compute z-scores


# Completed

figure out package structure for zwl.pyblp & symbols subpackage.

FX - get daily at time
resample tidy up
