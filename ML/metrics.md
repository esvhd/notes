# Feature Engineering

## Measuring Performance

### $R^2$

Cons:

1. It measures correlation, not accuracy. Could have strong linear relationship but do **not** conform to the 45-degree line of aggrement. E.g. under-predict one extreme, over-predicts the other extreme (tree-based ensemble methods are notorious for this).

**Tip:** Use Concordance Correlation Coefficient (CCC)

2. It can show very optimistic results when the outcome has large variance.

3. Can be misleading if there are a handlful of outcome values that are far away from the overall group of the observed and predicted values, i.e. different clusters.

**Tip:** distribution plot of predicited residuals can reveal skewed forecasts.
Also this allows us to compare models, the one with narrower distribution
of residuals may be better.

### Robust Metrics

1. When order matters more than actual value: **spearman rank correlation**
2. Median absolute deviation (MAD)
3. MAE

### Classification Metrics

In the two-class case, binomial **log-likelihood** has some advantage over
**Gini** and **entropy** metrics: Gini and entropy metrics **cannot** distinguish between very good and very bad models.

**Tip**: instead of focusing on yes/no type of hard class predictions and their metrics, focus on AUC statistics from ROC or PR curves FIRST. Once a good model is found, use ROC or PR curve to examine the appropriate probability cutoffs.

### Predictive Modeling Process

#### Resampling

K-fold, repeated K-fold, bootstrapping, monte-carlo CV, time series CV.

With $N$ repeats or K-fold CV, the variance in estimating average eval metric would decrease by $\sqrt{N}$.

As the no. of samples in the training set shrinks, the resampling estimate's bias **increases**. Therefore, 10-fold CV has lower bias than 5-fold CV.

In terms of variance: bootstramp < Repeated K-fold CV < simple K-fold CV.

What can be **outside** of resampling:

- centreing and scaling
- Small amount of imputation (but not large amount)

#### Comparing Models using Training Set

Process here is to use the same resamples across different models that are being evaluated.
Then, formal statistical inference can be done with the evaluation metrics from different models.

**Tip**: follow up trends discovered in exploratory data analysis with "stress testing" / resampling to get an objective sense of whether the trend is likely to be real.

## Engineering Numeric Features

### Kernel PCA

Traditional PCA is less effective in dimension reduction when the data should
be augmented with additional features to predict a response $y$.

Kernel PCA removes the need to create addtional terms in the feature matrix,
$X$.

One drawback is that it involves solving the eigenvalues for an $N \times N$
matrix where $N$ is the number of samples. This doesn't work well when dataset
is large. See Bishop's 2006 book.

### ICA

PCA components are orthogonal / uncorrelated, they may not be **statistically independent** of one another (i.e. they a covariance of 0).

ICA creates new components that are linear combinations of the original
variables, but in a way that the components are as **statistically independent**
from one anther as possible. There is no unique ordering of the components.

## Missing Data

For smaller data sets, use:

- Heatmap
- Co-occurance plots

For larger datasets, turn data matrix into indicator matrix (missing yes / no),
then run PCA and plot the first and second scores (components).

## Stats

From probability theory, we know that:

$$ Var[X-Y] = Var[X] + Var[Y] - 2 \times Cov[X, Y] $$

Wickham and Grolemund (2016): "If you think of variation as a phenomenon that creates uncertainty,
**covariance** is a phenomenon that reduces it."


