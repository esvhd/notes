# Feature Engineering

## Measuring Performance

### $R^2$

Cons:

1. It measures correlation, not accuracy. Could have strong linear relationship but do **not** conform to the 45-degree line of aggrement. E.g. under-predict one extreme, over-predicts the other extreme (tree-based ensemble methods are notorious for this).

**Tip: Use Concordance Correlation Coefficient (CCC)**

2. It can show very optimistic results when the outcome has large variance.

3. Can be misleading if there are a handlful of outcome values that are far away from the overall group of the observed and predicted values, i.e. different clusters.

**Tip: distribution plot of predicited residuals can reveal skewed forecasts.**



### Robust Metrics

1. When order matters more than actual value: **spearman rank correlation**
2. Median absolute deviation (MAD)
3. MAE
