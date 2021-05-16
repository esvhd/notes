# Notes on random stuff I read about...

<!-- MarkdownTOC levels='2,3' autolink="true" -->

- [Notes on random stuff I read about...](#notes-on-random-stuff-i-read-about)
  - [Heatmap Clustering](#heatmap-clustering)
  - [Clustering](#clustering)
  - [Hirearchical Agglomerative Clustering](#hirearchical-agglomerative-clustering)
    - [Accessment of Hierarchical Clustering](#accessment-of-hierarchical-clustering)
    - [Other Clustering Stuff](#other-clustering-stuff)
  - [Multi-Class Classification Scoring](#multi-class-classification-scoring)
    - [Confusion Matrix - `sklearn` format](#confusion-matrix---sklearn-format)
    - [Macro Averaging](#macro-averaging)
    - [Micro Averaging](#micro-averaging)
    - [Comparison](#comparison)
  - [Loss Functions](#loss-functions)
    - [Mean Square Error (MSE), L2 Loss](#mean-square-error-mse-l2-loss)
    - [Mean Absolute Error (MAE), L1 Loss](#mean-absolute-error-mae-l1-loss)
    - [Huber Loss](#huber-loss)
    - [Log-Cosh Loss](#log-cosh-loss)
    - [Quantile Loss](#quantile-loss)
  - [Causal Impact](#causal-impact)
    - [Bayesian Structural Time Series](#bayesian-structural-time-series)
  - [Bayesian Credible Region (CR) vs Frequentist Confidence Interval (CI)](#bayesian-credible-region-cr-vs-frequentist-confidence-interval-ci)
  - [Variance Estimations for Time Series with Autocorrelation](#variance-estimations-for-time-series-with-autocorrelation)
  - [RANSAC](#ransac)
  - [Statistics Done Wrong](#statistics-done-wrong)
  - [Time Series Prediction Tools](#time-series-prediction-tools)

<!-- /MarkdownTOC -->

## Heatmap Clustering

[post](http://nicolas.kruchten.com/content/2018/02/seriation/)

R package `seration`. Demo code [here](https://github.com/nicolaskruchten/seriation)

Talked about 3 methods:

1. Agglomerative clustering,
2. Optimal Leaf Ordering (starts with agglomerative clustering output then reorder branches of the dendrogram so as to minimize the sum of dissimilarities between adjacent leaves),
3. Traveling salesman (find the order of rows that minimizes the sum of dissmilarities, unconstrained by the clustering tree).

## Clustering

Some notes based on ESL.

Notes on Combinatorial algorithms for clustering, for the formula on page 508.

Equation `14.28` has three loops:

```
# this code computes within cluster average distance metric
# K clusters
dist = 0
for each cluster in K:
  for each data sample i in cluster k:
    for each data sample i' in cluster k:
      dist += distance(x_i, x_i')
dist = dist / 2
```

Equation `14.29` computes between cluster average distance:

```
# this code computes between cluster average distance metric
# K clusters
dist = 0
for each cluster in K:
  for each data sample i in cluster k:
    for each data sample i' NOT in cluster k:
      # computes distance between sample i and other samples not in its cluster
      dist += distance(x_i, x_i')
dist = dist / 2
```

**Total point** is the sum of within and between cluster distances.

## Hirearchical Agglomerative Clustering

**Single vs Complete**-link clustering

Good descriptions in [Information Retrival](https://nlp.stanford.edu/IR-book/html/htmledition/single-link-and-complete-link-clustering-1.html) book. Paragraphs from the book:

"In single-link clustering or single-linkage clustering , the similarity of two clusters is the similarity of their
most similar members (see Figure 17.3 (a)). This single-link merge criterion is **local**. We pay attention solely to
the area where the two clusters come closest to each other. Other, more distant parts of the cluster and the clusters'
overall structure are not taken into account."

"In complete-link clustering or complete-linkage clustering , the similarity of two clusters is the similarity of their
most dissimilar members (see Figure 17.3 (b)). This is equivalent to choosing the cluster pair whose merge has the
smallest diameter. This complete-link merge criterion is **non-local**; the entire structure of the clustering can
influence merge decisions. This results in a preference for compact clusters with small diameters over long, straggly
clusters, but also causes sensitivity to outliers. A single document far from the center can increase diameters of
candidate merge clusters dramatically and completely change the final clustering."

`scipy.cluster.hierarchy.linkage()` function has good documentation of this also.
Format of the output from `linkage()`:

- returns the linkage matrix `Z`
- each row is in the format of `[idx1, idx2, distance, num_sample_in_cluster]`
  where `idx1` & `idx2` are indices of the samples merged into a sub-cluster
- indices can be **more** than the no. of samples, in which case it starts to
  refer to `Z[i]`, i.e. merged clusters.

`single-linkage` defines distance between cluster $u$ and $v$ as:
$d(u, v) = min \big( dist(u[i], v[j]) \big)$ for all points $i$ in
cluster $u$ and $j$ in cluster $v$.

`complete-linkage` defines this distance as $d(u, v) = max\big( dist(u[i], v[j]) \big)$

Other methods also exists, such as `average`, `weighted`, `centroid`, `median` and `ward`.

ESL has a good section discussing the pros and cons of each method. Let's define
cluster diameter $D_G$ as the max dissimilarity within a cluster:

$$ D_G = max_{i \in G, i' \in G} d_{ii'} $$

- Single-linkage (SL)
  - tends to produce clusters with large diameter
  - can violate "compactness" property, members in the same group can be quite
    different to others in the same group.
- Complete-linkage (CL)
  - tends to produce compact groups with small diameter
  - can violate the "closeness" property - some observations in a group may
    look more like members of other clusters.
- Group Average
  - a compromise of the two above.
  - result is sensitive to the numerical scale on which dissmilarities are
    computed, wherease SL & CL results are invariant as long as the ranking
    of dissmiliarities are the same.
  - Nicer statistical consistency property.
- Ward: minimises total within-cluster variance.

### Accessment of Hierarchical Clustering

Cophenetic correlation coefficient can be used to asset how well the clustering
result matched the original distance matrix.

This can be done by first computing **Cophenetic distance** with
`scipy.cluster.hierarchy.cophenet` and then measure the (spearman) correlation
bewteen the upper triangle of the pairwise distance matrix and the
upper triangle of the Cophenetic distance matrix.

### Other Clustering Stuff

Great [blog](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/) that explains many
topics.

## Multi-Class Classification Scoring

[sklearn docs](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel)

Good example on [StackExchange](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001).

### Confusion Matrix - `sklearn` format

Actual vertically, Predicted horizontally.

|                   | Negative (Predicted) | Positive (Predicted) |
| ----------------- | -------------------- | -------------------- |
| Negative (Actual) | TN                   | FP                   |
| Positive (Actual) | FN                   | TP                   |

### Macro Averaging

Confusion matrix is computed for **each** class separately, each class has its own
metrics such as precision / recall / F1. Overall scores are simple / unweighted
averages of all class scores.

This can hide imbalanced class problems when the infrequent class is the more
important one.

### Micro Averaging

Confusion matrix is computed **globally for all classes**.

**Preferred** when correctly predicting the infrequent class is the task of interest.

### Comparison

With **macro** averaging, the confusion matrix of each class is computed in a
one-vs-other fashion. Then this is used to compute precision / recall / F1
score for each class separately first. The overall metrics are simple / unweighted
averages of the metrics for all classes.

In a multi-class setting, with **micro** averaging, confusion matrix is computed
for all classes at the same time. Hence, False Positive and False Negative
numbers are always the **same**, i.e. for a false negative in one class,
there is a false positive for another class.

Therefore, in a multi-class setting, **micro** averaging will always give the
**same** precision and recall. Better in imbalanced class problems than macro
averaging.

## Loss Functions

Good summary and comparison in this [post](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0).

### Mean Square Error (MSE), L2 Loss

Pros:

- differentiable, gradient is continuous, easier to train / optimize

Cons:

- Sensitive to outliers compare to MAE

### Mean Absolute Error (MAE), L1 Loss

Pros:

- More robust, less affected by outliers / noisy data

Cons:

- Gradient is the same for different points, harder to stablize as gradient remains large around minimum. Can use dynamic  learning rate to work around this, i.e. shrink learning rate around minimum.

**MSE** predicts the mean, **MAE** predicts the median of data.

### Huber Loss

Use L1 if loss greater than certain threshold `epsilon`, otherwise use L2.

Pros: This resolves the gradient issue for L1 loss, and the outlier issue for L2.

Cons: Additional parameter `epsilon` to tune.

### Log-Cosh Loss

$$ L(\hat{y}, y) = \sum^{N}_{i=1} \log\bigg[\cosh\big(\hat{y}_i - y_i \big) \bigg]$$

Pros:

* `log(cosh(x))` is approximately `(x ** 2) / 2` for small `x` and `np.abs(x) - np.log(2)` for large `x`. This makes it like MSE but less prone to outliers.

* Twice-differentiable, 2nd derivative is available, very useful for `XGBoost` which uses Newton's method.

Cons:

* Still suffers from constant gradient and hessian for large off-target predictions, resulting in **absent** of split in `XGBoost`.

### Quantile Loss

Like MAE loss, but assigns different weight depending on the value of prediction vs targets. E.g.

$$ L(\hat{y}, y) = \sum^{N}_{i = \hat{y}_i < y_i} (1 - \gamma) \times \mid \hat{y}_i - y \mid +  \sum^{N}_{i = \hat{y}_i \geq y_i} (\gamma - 1) \times \mid \hat{y}_i - y_i \mid$$

For $\gamma \in [0, 1]$.


## Causal Impact

[User guide](https://google.github.io/CausalImpact/CausalImpact.html#running-an-analysis)

Somes notes on Google's `CausalImpact` R package by Kay Brodersen et al.

The idea here is to use correlated time series unaffect by the event to
provide a counterfactual estimate (called **synthetic control**) of the
impacted time series.

This estimation is done by using a Bayesian structural time series model,
which is essentially a diffusion-regression state space model.

Comparing to traditional methods such as ARIMA, BSTS offers the following:

* not relying on differencing, lags or moving averages
* can visually inspect the underlying components of the model, e.g. trend,
seasonality, etc.
* posterior uncertainty of individual components.
* impose priors on the model coefficients
* ARIMA models can be expressed as BSTS models
* dynamic regression coefficients. In `CausalImpact` this is achieved by
pass additional parameters:`CausalImpact(..., model.args=list(dynamic.regression=TRUE))`

Traditional approaches such as `diff in diff` suffer from:

* can only model static relationships only
* requires observations to be i.i.d. - usually not practical for time series
* difference between treatment and control is constant (no trend)

To see which predictor variables were used in the model, run:

```
plot(impact$model$bsts.model, "coefficients")
```

Broadly speaking there are three sources of info available for constructing an adequate synthetic control:

1. time series behaviour of the reponse itself, prior to the intervention
(autocorrelation).

2. behaviour of other time series that were predictive of the target series prior to the intervention. (challenge is to pick the relevant subset to use as
a contemporaneous controls). Feature selection problem.

3. Bayesian framework, prior knowledge about the model parameters.

Approach here is to allow us to choose from a large set of potential controls
by placing a **spike-and-slab prior** on the set of regression
coefficients and by allowing the model to average over the set of controls.

Some **assumptions** made in this methodology:

* Assumes that covariates are unaffected by the effects of treatment. When
there are spill-over effects, the effect of the treatment would be
**underestimated**.

From Wikipedia [page](https://en.wikipedia.org/wiki/Spike-and-slab_variable_selection):

**Spike-and-slab regression** is a Bayesian variable selection technique that
is particularly useful when the number of possible predictors is larger than
the number of observations. In this context, it is used as a feature selection
tool - spike around 0 for coefficients, imposing prior that there is no
relationship, let the data speak for itself.

### Bayesian Structural Time Series

More details [here](http://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf), from the authors of R's `bsts`
package Steven L. Scott.

[Wikipedia](https://en.wikipedia.org/wiki/Bayesian_structural_time_series)

[blog on using `bsts`](https://multithreaded.stitchfix.com/blog/2016/04/21/forget-arima/) and [here](https://multithreaded.stitchfix.com/blog/2016/01/13/market-watch/) on `CausalImpact`.

Allows us to model **trend, seasonality and either static or dynamic regression
coefficients**.

A structural time series model can be described by a pair of equations:

$$
\begin{aligned}
y_t &= Z^T_t \alpha_t + \epsilon_t & \epsilon \sim \mathcal{N}(0, H_t) \\
\alpha_{t+1} &= T_t \alpha_t + R_t \eta_t & \eta_t \sim \mathcal{N}(0, Q_t)
\end{aligned}
$$

First equation above is called the **observation** equation, second is called
the **transition** equation.

$Z_t, T_t, R_t$ typicall contain a mix of known values (often 0 and 1), and
unknown parameters.

* $T_t$ is a square transition matrix
* $R_t$ can be rectangular if a portion of the state transition is
deterministic.
* $Q_t$ is a full rank variance matrix
$H_t$ is a positive scaler

Models that can be described by the two equations above are said to be in
**state space form**. ARIMA and VARMA models can be expressed in state space
form.

For paper from Scott above, an example model can be written as:

$$
\begin{aligned}
y_t &= \mu_t + \tau_t + \beta^T x_t + \epsilon_t \\
\mu_t &= \mu_{t-1} + \delta_{t-1} + u_t \\
\delta_t &= \delta_{t-1} + v_t \\
\tau_t &= -\sum^{S-2}_{s=0} \big( \tau_{t-s} \big) + w_t
\end{aligned}
$$

where:

* $\eta_t = (u_t, v_t, w_t)$ contains independent components of Gaussian
random noise, $N(0, \sigma^2_e)$

* The seasonal component $\tau$ can be thought of as a set of $S$
dummy variables with dynamic coefficients constrained to have zero expectation
over a full cycle of $S$ seasons

* $\mu_t$ is current level of the trend

* $\delta_t$ is current slope of the trend

Also there is detailed discussion on seasonality in Brodersen's
[paper](https://ai.google/research/pubs/pub41854), section 2.1.

## Bayesian Credible Region (CR) vs Frequentist Confidence Interval (CI)

A great blog post by @jakevdp that had a more interesting comment section on
this topic [here](http://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/).
And an excellent [stackoverflow answer](https://stats.stackexchange.com/questions/2272/whats-the-difference-between-a-confidence-interval-and-a-credible-interval/2287#2287)

Key point is the with CI, frequentists provide the right answer to the
**wrong** question.

Typically from the data given, we are interested in what the **given data**
tells us. That's what Bayesian CR tells us.

Frequentist CI tells us, if you repeatedly see **data of this kind**, there is
$X$% chance that the true value of $\theta$ falls inside of the CI.

But we are not interested in data of this kind, we are interested in what this
piece of data tells us!

## Variance Estimations for Time Series with Autocorrelation

For a data set with $T$ observations, $\rho_l$ is the autocorrelation with $l$-lag, we have:

$$ var(\hat{x}) = \bigg[ \frac{T + 2 \sum_{t=1}^T (T - l) \rho_l}{T} \bigg] \frac{1}{T} var(x_t) $$

## RANSAC

`sklearn`'s doc has links to some [papers](https://scikit-learn.org/stable/modules/linear_model.html#ransac-regression).

The user guide does a reasonable job explaining the algo.
At high level, it's a **non-deterministic** algo that repeatedly samples a subset
of the full data, fits a model and compare predicted values vs true lables,
those samples with error below `residual_threshold` are marked **inliers**,
those above are marked as **outliers**. The final model is the one that has
the most **inlier** samples.

An example comparing 3 robust linear methods [here](https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py)
is very good.

Takeaway from this page is that RANSAC is good for strong outliers in the `y` direction.
Other methods to consider are:

- `TheilSenRegressor`: good for small outliers in both `X` and `y`, but beyond certain point it does worse than `OLS`.
- `HuberRegressor`: Cannot compare scores directly with others. Does not ignore outliers, only lessens their effect.

## Statistics Done Wrong

Use power analysis when designing the experiment / research. Statistical power is affected by three things:

- The size of the effect one is looking for
- The sample size
- Measurement error

Use **assurance** instead of power when you need to measure an effect with precision.

Use confidence intervals instead of just p-values.

Adjust for confounding factors.

Watch out for **pseudoreplication**, such as taking separate measurements of the same subject over time (autocorrelation). Example, 1970 study claiming women's menstrual cycles can synchronise if they lived in close contact.

Use methods such as **hierarchical models** and **clustered standard errors** to account for strong dependence between your measurements.

p-value and base rate fallacy, i.e. detecting rare events. p-value tells us the probability of being surprised by data assuming no effect. It does **not** tell us the chance of the hypothesis is true.

**Bonferroni** percedure implicitly assumes that **every** null hypothesis tested in multiple comparison is true. Use **Benjamini-Hochberg** procedure instead for multiple testing:

1. Perform statistical tests and get p-value for each experiment. Make a list and sort it in ascending order.
2. Choose a **false-discovery** rate and call it `q`. Call the number of statistical tests `m`.
3. Find the largest `p` value such that $p \leq iq/m$ where $i$ is the p-value's order in the sorted list
4. call that p-value and all smaller than it statistically significant.

This procedure guarantees that out of all statistically significant results, **on average no more than $q$ percent will be false positives.**

**Error bars** could represent three different things:

1. 2x standard deviation of the measurement, 1x each side. Measures the spread of data.
2. 95% confidence interval for the estimate
3. 2x standard error for the estimate, 1x each side.

2 & 3 both estimate how far the average of this sample might be from the **true** average.

Non-overlapping standard errors do not suggest the difference between the two is **not** statistically significant.

Standard deviation do not give enough information to judge significance, whether they overlap or not.

Don't arbitrarily split continuous variables into discrete groups unless you have good reasons.

Don't choose the groups to maximise statistical significance.

Also some bad examples using step-wise regression, which is essentially iteratively testing for parameter significance... Problems: multiple comparison, bound to produce false positives, no guarantees about the overall false positive rate, nor are they guaranteed to select the best features.

Use random assignment to eliminate confounding variables / Simpson's paradox (some effect is declared by when controlling for some variable the effect can no longer be found, Berkerley female application acceptance rate example).

## Time Series Prediction Tools

- `sktime`
- `pytorch-forecast`
- FaceBook `Prophet`
- `tslearn`
- `tsfresh`

Great list [here on Quora](https://www.quora.com/What-is-the-most-useful-Python-library-for-time-series-and-forecasting).
