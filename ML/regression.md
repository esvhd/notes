# Regression / Stats Notes

For OLS regression problems, assume model $y = X\beta + \varepsilon$, we have solution for $\beta$:

$$
\begin{aligned}
\ \hat{\beta} &= (X^{T}X)^{-1}X^{T}y
\end{aligned}
$$

Let $\hat{y}$ be the model predicted value of $y$.

## Regression Metrics

### **RSS**, Residual Sum of Squares, or sometime known as Sum of Square Erros, **SSE**

$$RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}e_i^{2}$$

### **MSE**, Mean Sequare Error, or **RSE**, Residual Standard Error

For $X$ in the dimention of $(n, p)$, i.e. $n$ observations, $p$ features (degree of freedom = $p + 1$, assuming there is an intercept):

$$MSE = \frac{RSS}{n - p - 1}$$

$$RSE = RMSE = \sqrt{MSE}$$

**R code**, degree of freedum is given by: `summary(model)$sigma`

### **TSS**, Total Sum of Squares

$$\bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_i$$

$$TSS = \sum_{i=1}^{n}(y_i - \bar{y}_i)^2$$

### $R^2$, Adjusted $R^2$

$$
\begin{aligned}
\ R^2 &= \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS} \\
\ \\
\ Adjusted.R^2 &= 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}
\end{aligned}
$$


### F-test of significance of all parameters. ISLR, p77

Look up for p-value in F-table for degree of freedum of $(p, n-p-1)$ (columns, rows).

NULL hypothesis $H_0$ is that all parameters are zero.

$$F = \frac{(TSS-RSS) / p}{RSS / (n-p-1)}$$

### F-test of two models

Consider a large model of $p$ features, and a smaller model of a subset of $p$ features, say $q$ features. F-test can be used to see which model is better.

NULL hypothesis $H_0$ is that the smaller model is better.

$$F = \frac{(RSS_q - RSS_p)/q}{RSS_p / (n-p-1)}$$


### $R^2$ and Correlation

$$
\begin{aligned}
\ R^2 =
\begin{cases}
p = 1, \text{corr($y$, $x$)^2} \\
p > 1, \text{corr($y$, $\hat{y}$)^2} \\
\end{cases}
\end{aligned}
$$

## Model selection Metrics for p > 1

Model selection should be done using metrics on **test data, not training data**.

### $C_{p}$, choose model with lowest value

$$C_{p} = \frac{1}{n}(RSS + 2p\hat{\sigma}^2)$$

Where $\hat{\sigma}^2$ is the estimated residual variance.


### AIC, lower value better

$$AIC = \frac{1}{n\hat{\sigma}^2}(RSS + 2p\hat{\sigma}^2)$$

### BIC, places heavier penalty on models with higher dimensions.

$$BIC = \frac{1}{n}(RSS + \log{(n)}p\hat{\sigma}^2)$$

##  Information Theory and Deviance

(Statistical Rethinking)

**Information Entropy**: for $n$ possible events and each event $i$ has probability $p_i$, then the unique measure of uncertainty is:

$$ H(p) = -\mathbb{E}[log(p_i)] = - \sum_{i=1}^{n}p_{i} log(p_{i}) $$

**Kullback-Leibler (K-L) Divergence**: the divergence is the average difference in log probability between the target ($p$) and model ($q$). Where $p$ and $q$ are parameters of the **true model** and our estimated model respectively. In otherwords, it is defined as the **additional** entropy induced by using $q$.

$$ D_{KL} = \sum p_{i}[log(p_i) - log(q_i)] = \sum_{i} p_i log(\frac{p_i}{q_i}) $$

**Cross Entropy**: Using a probability distribution $q$ to predict events from another distribution $p$:

$$
\begin{aligned}
\ H(p, q) &= - \sum_{i}p_i log(q_i) \\
\ D_{KL}(p, q) &= H(p, q) - H(p) \\
\ &= - \sum_{i=1}^{n}p_{i} log(q_{i}) - (- \sum_{i=1}^{n}p_{i} log(p_{i})) \\
\ &= - \sum_{i=1}^{n}p_{i} (log(q_{i}) - log(p_{i}))
\end{aligned}
$$

However, in real world, the true model, $p$, is usually unknown. Therefore **Deviance** is defined as:

$$ D(q) = - 2\sum_{i} log(q_i) $$

`R` function `-2 * logLik()`

**AIC** provides an estimate of average out-of-sample deviance:

$$ AIC = D_{train} + 2p $$

where $p$ is the number of free parameters to be estimated in the model.

** Deviance Information Criterion (DIC) **

** Widely Applicable Information Criterion **



## Accurancy of Sample Mean $\hat{\mu}$

$$
\begin{aligned}
\ \hat{SE}(\hat{\mu}) &= \frac{\sigma^{2}}{n} \\
\ \hat{\mu} &= \frac{1}{n}\sum_{i=1}^{n}x_i \\
\ \sigma^2 &= variance(x)
\end{aligned}
$$

## Hierarchical Principal, ISLR, p89

If we include an interaction in a model, we should also include the main effects, even if the p-value associated with their coefficient are not significant. E.g. if $X_1 \times X_2$ seems important, the model should include both $X_1$ and $X_2$.

## Confidence Intervals, p = 1

CI - Confidience Interval of model prediction **on average**, where $\alpha$ is the probability, e.g. 95%:

$$CI=\hat{y}_n \pm t_{(\alpha/2, n-p-1)} \times \sqrt{MSE * (\frac{1}{n} + \frac{(x_h - \bar{x})^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2})}$$

$t_{(\alpha/2, n-p-1)}$ is based on Normal distrubtion table.

Conditions for CI

1. $x_h$ is within the range of training data, i.e. within scope of the model
2. **LINE**: Linear, Independent errors, normal errors, equal variance. Still works if error is approximately Normal. Or, if $n$ is large, error can deviate substantially from normal.

## Prediction Interval, p = 1

PI - Prediction Interval, for point predictions, where $\alpha$ is the probability.

$$PI = \hat{y}_h \pm t_{(\alpha/2, n-p-1)} \times \sqrt{MSE \times (1 + \frac{1}{n} + \frac{(x_h - \bar{x})^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2})}$$

$t_{(\alpha/2, n-p-1)}$ is based on Normal distrubtion table.

Conditions for PI

1. $x_h$ is within the range of training data, i.e. within scope of the model
2. **LINE**: Linear, Independent errors, normal errors, equal variance. Strongly depends on errors being normal.

## Confidence & Prediction Intervals where p > 1

Reference:

* https://onlinecourses.science.psu.edu/stat501/node/314

* https://onlinecourses.science.psu.edu/stat501/node/315

Let $X_n = (1, x_{n,1}, x_{n, 2}, \ldots, x_{n, p})^T$, $X_h$ is an observation.
$$
\begin{aligned}
\ CI &= \hat{y}_h \pm t_{(\alpha/2, n-p-1)} \times SE(\hat{y}_h) \\
\ \\
\ PI &= \hat{y}_h \pm t_{(\alpha/2, n-p-1)} \times \sqrt{MSE + SE(\hat{y}_h)^2} \\
\ \\
\ SE(\hat{y}_h)^2 &= MSE \times X_{h}^T(X^{T}X)^{-1}X_{h} \\
\end{aligned}
$$


### Python
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLSResults.html

Confidence Interval: ```OLSResults.conf_int()```

Prediction Interval:
http://markthegraph.blogspot.de/2015/05/using-python-statsmodels-for-ols-linear.html

```{python}
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(re)
```

See source:
https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/regression/predstd.py

### R

`predict` or `predict.lm`. check out `help(predict.lm)` and see `internal` parameter.

## Variance of OLS $\hat{\beta}$

Let:

$$ \sigma^2 = MSE = RSE^2 $$

### I.I.D Errors

Based on OLS model formula above, subsituting in for $y$:

$$
\begin{aligned}
\ \hat{\beta} & = (X^{T}X)^{-1}X^{T}y \\
\ &= (X^{T}X)^{-1}X^{T}(X\beta + \varepsilon) \\
\ &= (X^{T}X)^{-1}X^{T}X\beta + (X^{T}X)^{-1}X^{T}\varepsilon \\
\ &= \beta + (X^{T}X)^{-1}X^{T}\varepsilon\\
\ \\
\ var(\hat{\beta}) &= \mathbb{E}[(X^{T}X)^{-1}X^{T}\varepsilon\varepsilon^{T}X(X^{T}X)^{-1}] \\
\ &= (X^{T}X)^{-1}\mathbb{E}[X^{T}\varepsilon\varepsilon^{T}X](X^{T}X)^{-1}\\
\ &= (X^{T}X)^{-1}X^{T}\mathbb{E}[\varepsilon\varepsilon^{T}]X(X^{T}X)^{-1}\\
\ \because \varepsilon &\sim \mathcal{N}(0, \sigma^2) \\
\ \therefore \mathbb{E}[\varepsilon\varepsilon^{T} \mid X] &= \Omega = \sigma^2 I \\
\ &= (X^{T}X)^{-1}X^{T}\Omega X(X^{T}X)^{-1}\\
\ &= \sigma^{2}(X^{T}X)^{-1}X^{T}X(X^{T}X)^{-1}\\
\ var(\hat{\beta}) &= \sigma^{2}(X^{T}X)^{-1}
\end{aligned}
$$

### I.N.I.D Errors (Heteroskedasticity)

To deal with uncorrelationed residuals

$$
\begin{aligned}
\ var(\hat{\beta}) &= (X^{T}X)^{-1}X^{T}X\mathbb{E}[\varepsilon\varepsilon^{T}](X^{T}X)^{-1}\\
\ \because \mathbb{E}[\varepsilon\varepsilon^{T}] &= \sigma^{2}\Omega \\
\ var(\hat{\beta}) &= (X^{T}X)^{-1}X^{T}\sigma^{2}\Omega X(X^{T}X)^{-1}\\
\ var(\hat{\beta}) &= \sigma^{2}(X^{T}X)^{-1}X^{T}\Omega X(X^{T}X)^{-1}\\
\ \therefore \hat{\beta} &\sim \mathcal{N}(\beta, \sigma^{2}(X^{T}X)^{-1}X^{T}\Omega X(X^{T}X)^{-1})
\end{aligned}
$$

### N.I.N.I.D Errors (HAC)

To deal with correlated residuals, use Newey West. See video below for details and Greene's book, p517 - p518.

https://www.youtube.com/watch?v=HznGehi6xNQ

**Newey West essentially has a modified / weighted variance-covariance matrix $\Omega_{NW}$.**

Essentially applying weights $\omega$ diagonally to $\Omega$, example:

$$
\Omega_{NW} =
 \begin{pmatrix}
  e_{1}e_{1} & \omega_{1}e_{1}e_{2} & \omega_{2}e_{1}e_{3} & 0 & \cdots & 0 \\
  \omega_{1}e_{1}e_{2} & e_{2}e_{2} & \omega_{1}e_{2}e_{3} & \omega_{2}e_{2}e_{4} & \cdots & 0 \\
  \omega{2}e_{1}e_{3} & \omega_{1}e_{2}e_{3} & e_{3}e_{3} & \omega_{1}e_{3}e_{4} & \cdots & 0 \\
  0 & \omega_{2}e_{2}e_{4} & \omega_{1}e_{3}e_{4} & e_{4}e_{4} & \cdots & 0 \\
  \vdots  & \vdots  & \vdots & \vdots & \ddots & \vdots  \\
  0 & 0 & 0 & \cdots &\omega_{1}e_{n-1}e_{n} & e_{n}e_{n}
 \end{pmatrix}
$$

With $0 < \omega_1 < \omega_2 < \cdots < \omega_n < 1$


### Newey West

http://stackoverflow.com/questions/23420454/newey-west-standard-errors-for-ols-in-python

In `R`, `NeweyWest()` in the `sandwich` package use Newey West 1994 paper to automatically select the lag.

In `Python`, currently there isn't a way to automatically select the lag. Some python packages, such as `arch`, use a default value of $4(n/100)^{2/9}$ where $n$ is the length of data, i.e. nobs.

**`R` code to test $\hat{\beta} == 0$ with Newey West vcov matrix for correlated residuals**:

```{R}
library(lmtest)
library(sandwich)

lm.fit <- lm(y~x)
coeftest(lm.fit, vcov. = NeweyWest)

```

Or manually:

```{R}
lm.fit <- lm(y~x)
# IID case
# std_err <- sqrt(diag(vcov(lm.fit)))
std_err <- sqrt(diag(NeweyWest(lm.fit)))
tstat <- coef(lm.fit) / std_err
p_vals <- pt(abs(tstat), df=df.residuals(lm.fit), lower.tail=FALSE)
```

** `R` code to get $var(\hat{\beta})$ **:

```
var_beta <- vcov(lm.fit)
# or
x <- lm.fit.matrix(~V1+V2, data=df)
var_beta <- summary(lm.fit)$sigma^2 * solve(t(x) %*% x)
```

**`Python` code for Newey West**:

```{python}
ols = sm.ols(...).fit(cov_type='HAC',cov_kwds={'maxlags':1})
ols.summary()
# or
ols = sm.ols(...).fit()
ols2 = ols.get_robustcov_results(cov_type='HAC',maxlags=1)
ols2.summary()
```

## Ridge Regression

Formulation $y = X\beta + \lambda I \beta^{T}\beta + \varepsilon$.

OLS should not be used when $n < p$, i.e. when nobs < no. of features.

Generally when the relationship is close to linear, least square methods may have **high variance and low bias**. **Ridge regression outperforms when OLS estimates have high variance, or when $p > n$ as OLS cannot be used.**

Unlike OLS, whose coefficents are **scale equivariant**, ridge regression coefficients can vary significantly when variables are scaled. **Therefore, it is best to apply ridge regresion after standardising the predictors to have standard deviation of 1**, see ISRL p217:

$$ \tilde{x}_{ij} = \frac{x_{ij}}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\bar{x}_{j})}}$$

Derivation of $\hat{\beta}$:

$$
\begin{aligned}
\ RSS &= (y - X\beta)^{T}(y - X\beta) + \lambda I \beta^{T} \beta \\
\ &= y^{T}y - y^{T}X\beta - \beta^{T}X^{T}y + \beta^{T}X^{T}X\beta + \lambda I \beta^{T} \beta \\
\frac{\partial{RSS}}{\partial{\beta}} &= 0 - y^{T}X - X^{T}y + 2X^{T}X\beta + 2\lambda I \beta  = 0\\
\therefore (2X^{T}X + 2\lambda I)\beta &= 2X^{T}y \\
\ (X^{T}X + \lambda I)\beta &= X^{T}y \\
\ \beta &= (X^{T}X + \lambda I)^{-1}X^{T}y \\
\ \\
\ var(\hat{\beta}) & = \sigma^{2}\mathbb{W}(X^{T}X)\mathbb{W} \\
\ \mathbb{W} &= (X^{T}X + \lambda I)^{-1} \\
\ Bias(\hat{\beta}) &= -\lambda \mathbb{W}\beta \\
\end{aligned}
$$

Variance formula & useful links:

http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/9-1.pdf

http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf

https://onlinecourses.science.psu.edu/stat857/node/155

### Confidence & Prediction Intervals

See:

http://stats.stackexchange.com/questions/13329/calculate-prediction-interval-for-ridge-regression

http://stackoverflow.com/questions/39750965/confidence-intervals-for-ridge-regression

Due to the ridge being a biased estimator, Bootstrapping seems to be the best way to estimate prediction intervals here.

## Ridge vs Laso, ISRL p224

One does not dominate the other, use CV in practice to see which works better, depends on the true relationship in data.

Ridge shrinks parameters by the same proportion.

Lasso srhinks each OLS parameter by $\lambda / 2$. Therefore any parameter less than $\lambda / 2$ is shrunken to zero. This is known as **soft thresholding**.

# Linear Regression Diagnostics in `R`

## LINE Conditions

Reference: **R in Action**, summary of functions on p187.

In `R`, you can examine **LINE** with `plot()`:

```{R}
fit <- lm(weight ~ height, data=women)
par(mfrow=c(2, 2))
plo(fit)
```

Many functions in `R`'s `car` package can help:

* **L**: Linearity. Look at **Risidual vs Fitted values** plot, residual should be random, should not exhibit patterns.
    * **Component plus residual plots** (aka partial residual plots), `crPlots()`.
    * solution: transform **feature variables**, try `car::boxTidwell()` suggested power transforms. check for p-values for the variable to see if transform is needed.

* **I**: Independence of errors. Can't tell from the plots.
    * `durbinWatsonTest()`

* **N**: Normality. Look at Q-Q plot.
    * `qqPlot()`, plots the studentized residuals (aka studentized deleted residuals or jackknifed residuals) against a t-distribution with `n-p-2` degrees of freedom. (n - nobs, p - no. of features excluding intercept).
    * `residplot()` in **R in Action** on page 189.
    * solution: Transform **feature variables**. Use `car::powerTransform()`, check p-value for lambda(1) which has no transform to see if a transform is in fact needed.

* **E**: Equal variance / Homoscedasticity. Scale-location plot, sqrt(standardized residual) vs fitted values.
    * `ncvTest()`, NULL hypothesis is constant variance. A significant result would support heterscedasticity.
    * `spreadLevelPlot()`
    * Counter-measure is to transform the **response** with `log()` or `sqrt()`.
* **Global Test** package `gvlma`

## Multicollinearity

Use **Variance Inflation Factor**, `vif()` function in `car` package. General rules is $\sqrt{vif} > 2$ indicates a multicollinearity problem (R in Action).

See ISRL p101-102:

VIF is the ratio of the variance $\hat{\beta}_j$ when fiting the full model divided by the variance of $\hat{\beta}_j$ if fit on its own. Minimum of VIF is 1. **VIF > 5 or 10 indicates problems.**

$$ VIF(\hat{\beta}_{j}) = \frac{1}{1 - R^{2}_{X_{j} \mid X_{-j}}} $$

$R^{2}_{X_{j} \mid X_{-j}}$ is the $R^2$ from a regression of $X_j$ onto all other features. if $R^{2}_{X_{j} \mid X_{-j}}$ is close to 1, collinearity is present.

Two ways to correct for Multicollinearity:
* drop one of the problematic features
* combine the collinear features into one single feature.

## Unusual Observations

### Outliers

`outlierTest()` - Bonferroni adjusted p-value for the largest absolute studentized residual. **The test needs to be repeated if the largest data point is removed to check for other outliers.**

### High Leverage Points

These are observations with unusual combination of predictor values, i.e. outliers with regards to other predictors.

Compute **leverage statistics (aka. hat statistics)**, `hatvalues()` (**R in Action** p195, also see code plot that uses `identify()` function).

In general a hat statistics greater than 0.2 or 0.3 should be examined.

Formula from **An Introductino to Statistical Learning with Applications in R, (ISLR)**, p98:

$$ h_i = \frac{1}{n} + \frac{(x - \bar{x})^2}{\sum_{i^{'}=1}^{n}(x_{i^{'}} - \bar{x})^2} $$

$1/n < h_i < 1$, average leverage of all observations is $(p+1)/n$. Any observations with leverage statistics greatly exceeds the average should be checked.

In matrix form this is $H = X(X^{'}X)^{-1}X^{'}$, where $X$ is the design matrix, $h_{ii} = H_{ii}$ for $i^{th}$ data point.

https://en.wikipedia.org/wiki/Leverage_(statistics)

### Influential Observations

These have disproportional impact on the values of the model parameters. Identified by:
* **Cook's distance (D stat)** greater than $4/(n-p-1)$. In `R`: `plot(fit, which=4, cook.level=cutoff)`
* **Added variable plots**, `avPlots(fit, ask=FALSE, id.method='identify')` in `car`

**An outlier that also has high leverage is particularly dangerous.** See **ISLR** p99.

### Influence Plot

Combines all the above into `car`'s `influencePlot(fit, id.method='identify')`.

## Model Selection

* Stepwise regression, `MASS::stepAIC()`
* All subsets regression, `leaps::regsubsets()`
* Cross validation, `bootstrap::crossval()`

## Statistical Tests

ANOVA analysis is sensitive to outliers. Run `outlierTest()` before the analysis.

### Equality of Variances

* Barlett's test: `barlett.test()`
* Fligner-Killeen: `fligner.test()`
* Brown-Frosythe test: `HH::hov()`



## Power Analysis

**Power** is defined as: $1 - Prob(\text{Type II Error})$, i.e. it is the probability of finding evidence to reject the NULL hypothesis (finding an effect is there).

`pwr` package provides a list of power analysis tests. List of functions in **R in Action** page 242.

Useful ones:

* t-test: `pwr.t.test()`
* correlation: `pwr.r.test()`
* linear models: `pwr.f2.test()`

# Permutation Test & Bootstrap

Packages:

* `coin`
* `lmPerm`
* `logregperm` permutation test for logistic regression
* `glmperm` for GLM
* `boot`
