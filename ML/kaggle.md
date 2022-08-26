# Kaggle Competition Notes

## Web Traffic Time Series

1st Place Solution [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795)

Some quick notes:

* Seq2Seq model with cuDNN cells.
* Used lagged inputs as features, as effective as complicated attension.
* **COCOB optimizer** - predicts optimal learning rate for every training step
* Symmetric mean absolute percentage error loss [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
* Tried activation regularization but had only very slight improvement.
* Training data split:
    1. walk-forward split - wastes some data at the end of the dataset (reserved for validation)
    2. side-by-side split - time series data has strong autocorrelation. Model performance
    on validation dataset is trongly correlated to performance on training dataset, almost
    **uncorrelated** ot the actual model performance in the future. (not useful here)
* Used [SMAC3](https://automl.github.io/SMAC3/stable/) package for hyperparameter tuning.

To eeducing variance:

* Used averaging SGD (ASGD) to reduce variance.
* Set early spotting region empirically and used checkpoints in this region.
* Trained 3 models on different seeds.

Author used walk-forward split for model tuning, then tuned model without validation.

2nd Place Solution [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/44729) by @CPMP

Simple NN model, code [here](https://github.com/jfpuget/Kaggle/tree/master/WebTrafficPrediction)

3rd Place Solutin [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39367#223730) by @thousandvoices

Very interesting approach.

4th Place Solution by @leecming - FC network with 120-day lookback.

## Kaggle Oliver Realised Vol

### 1st place solution

<https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970>

**define Adversarial Validation**
Basically like what Jeremy Howard mentioned, label training and test data, train a classifier to see if you can identify the correct class for each sample. If you can (e.g. ROC > 0.5), then we may have regime shift in data. This is likely to be caused by certain features, so find the features that have the largest contribution to ROC, then consider excluding them.

**define Nearest neighbour feature aggregation**
Conceptually, the author used KNN to find the nearest other samples of the same stock that had similar trading environment, e.g. return / volume / realised vol, or potentially time of day. Think author created a feature that is the average realised vol for these neighbours.

Why recovering time-id with t-SNE type of dimension reduction was useful? See replies on other techniques from 3rd place solution

Model - ensemble of LightGBM / 1d-CNN / MLP

What did not work:

* TabNet took too long to train
* domino specific features
* dimension reduction on features

Did not try:

* LSTM / RNN / Auto encoder
* breakdown 10 mins into 5-min halves, use as meta-feature or data augmentation
