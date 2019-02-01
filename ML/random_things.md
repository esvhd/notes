# Notes on random stuff I read about...


- [Notes on random stuff I read about...](#notes-on-random-stuff-i-read-about)
  - [Heatmap Clustering](#heatmap-clustering)
  - [Permutation Importance for Random Forest Feature Importance](#permutation-importance-for-random-forest-feature-importance)
  - [Loss Functions](#loss-functions)
    - [Mean Square Error (MSE), L2 Loss](#mean-square-error-mse-l2-loss)
    - [Mean Absolute Error (MAE), L1 Loss](#mean-absolute-error-mae-l1-loss)
    - [Huber Loss](#huber-loss)
    - [Log-Cosh Loss](#log-cosh-loss)
    - [Quantile Loss](#quantile-loss)
  - [Causal Impact](#causal-impact)
    - [Bayesian Structural Time Series](#bayesian-structural-time-series)
  - [Bayesian Credible Region (CR) vs Frequentist Confidence Interval (CI)](#bayesian-credible-region-cr-vs-frequentist-confidence-interval-ci)


## Heatmap Clustering

[post](http://nicolas.kruchten.com/content/2018/02/seriation/)

R package `seration`. Demo code [here](https://github.com/nicolaskruchten/seriation)

Talked about 3 methods:

1. Agglomerative clustering,
2. Optimal Leaf Ordering (starts with agglomerative clustering output then reorder branches of the dendrogram so as to minimize the sum of dissimilarities between adjacent leaves),
3. Traveling salesman (find the order of rows that minimizes the sum of dissmilarities, unconstrained by the clustering tree).


## Permutation Importance for Random Forest Feature Importance

See this [post](http://parrt.cs.usfca.edu/doc/rf-importance/index.html), [github](https://github.com/parrt/random-forest-importances)

`sklearn` Random Forest feature importance and `R`'s default Randome Forest feature importance strategies are **biased**.

Solution is to compute **permutation importance** from Breiman and Cutler. Existing packages:

* Python: `rfpimp` through `pip`
* R: use `importance=T` in random forest constructor then `type=1` and `scale=F` in `R`'s `importance()` function.

Feature importance will only be reliable **if your model is trained with suitable hyper-parameters**.

Permutation importance works for all models, not just random forests. The procedure is as follows:

1. Train the model as usual
2. Record a baseline: score the model by passing a validation or test set.
3. For each feature (columns), permute the column values, compute the same score.
4. The importance of a feature is **the difference of scores between the baseline and the drop in score after permutation**.

The importance metrics here are **not** normalized and do not sum to 1. The specific values of importance do not matter, what matters is the **relative predictive strength**, i.e. ranking.

A more direct and accurate strategy is the **drop-column importance**. This requires establishing a baseline, and then drop a feature column and **re-train** the model. Clearly, this is more computationally intensive. The importance measure is the drop of score from the baseline, as before.

Here's the code snippet from the post.

```
import numpy as np

def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I
```


## Loss Functions

Good summary and comparison in this [post](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0).

### Mean Square Error (MSE), L2 Loss

Pros:
* differentiable, gradient is continuous, easier to train / optimize

Cons:
* Sensitive to outliers compare to MAE

### Mean Absolute Error (MAE), L1 Loss

Pros:
* More robust, less affected by outliers / noisy data

Cons:
* Gradient is the same for different points, harder to stablize as gradient remains large around minimum. Can use dynamic  learning rate to work around this, i.e. shrink learning rate around minimum.

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

Typicall from the data given, we are interested in what the **given data**
tells us. That's what Bayesian CR tells us.

Frequentist CI tells us, if you repeatedly see **data of this kind**, there is
$X%$ chance that the true value of $\theta$ falls inside of the CI.

But we are not interested in data of this kind, we are interested in what this
piece of data tells us!
