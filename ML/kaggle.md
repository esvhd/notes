# Kaggle Competition Notes

# Web Traffic Time Series

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
