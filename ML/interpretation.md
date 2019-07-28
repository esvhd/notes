# Notes on Feature Importance and Interpretation

<!-- MarkdownTOC levels='1,2,3' autolink="true" -->

- [Notes on Feature Importance and Interpretation](#notes-on-feature-importance-and-interpretation)
  - [Permutation Importance for Random Forest Feature Importance](#permutation-importance-for-random-forest-feature-importance)
  - [Partial Dependency Plots (PDP)](#partial-dependency-plots-pdp)
  - [Individual Conditional Expectation (ICE) Plots](#individual-conditional-expectation-ice-plots)
  - [Stratified Partial Dependence (StratPD)](#stratified-partial-dependence-stratpd)
    - [Paper Notation](#paper-notation)
    - [Stratification](#stratification)
    - [Experiments](#experiments)
    - [Noise in Data](#noise-in-data)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [TODO:](#todo)

<!-- /MarkdownTOC -->

## Permutation Importance for Random Forest Feature Importance

See this [post](http://parrt.cs.usfca.edu/doc/rf-importance/index.html), [github](https://github.com/parrt/random-forest-importances)

`sklearn` Random Forest feature importance and `R`'s default Randome Forest feature importance strategies are **biased**.

Solution is to compute **permutation importance** from Breiman and Cutler. Existing packages:

- Python: `rfpimp` through `pip`
- R: use `importance=T` in random forest constructor then `type=1` and `scale=F` in `R`'s `importance()` function.

Feature importance will only be reliable **if your model is trained with suitable hyper-parameters**.

Permutation importance works for all models, not just random forests. The procedure is as follows:

1. Train the model as usual
2. Record a baseline: score the model by passing a validation or test set.
3. For each feature (columns), permute the column values, compute the same score.
4. The importance of a feature is **the difference of scores between the baseline and the drop in score after permutation**.

The importance metrics here are **not** normalized and do not sum to 1. The specific values of importance do not matter, what matters is the **relative predictive strength**, i.e. ranking.

A more direct and accurate strategy is the **drop-column importance**. This requires establishing a baseline, and then drop a feature column and **re-train** the model. Clearly, this is more computationally intensive. The importance measure is the drop of score from the baseline, as before.

Here's the code snippet from the post.

```{python}
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

## Partial Dependency Plots (PDP)

Based on **ESL** p369-370.

Definitions: let feature space be $X \in \mathbb{R}^{N \times p}$, let $X_S \in \mathbb{R}^{N \times l}$ where $l < p$ be a subset of features. Let $C$ be the **complement** set, so $S \cup C = \{1, 2, \cdots, p\}$.

One way to define **average** or **partial** dependence of $f(X)$ on $X_S$ is:

$$ f_S (X_S) = \mathbb{E}_{X_C} f(X_S, X_C) $$

What this means is that we compute the following:

$$ \bar{f}_S (X_S) = \frac{1}{N} \sum_{i=1}^{N} f(X_S, x^{(i)}_C) $$

where $x^{(i)}_C$ is the $i$th example out of $N$ observations in the complement $C$ set, i.e. use $X_S$ and $x^{(i)}_C$ to form a new dataset (for features in $C$ we use values of $x^{(i)}_C$), then pass through the model, repeat $N$ times (using each of the $N$ examples), then take the average.

Alternatively, for each unique value of $X_S$, create a new dataset with $X_C$, run prediction and take average.

This is computationally intensive. Fortunately, for trees, this can be done efficiently without reference to the data.


Jeremy Howard showed some interesting python packages for interpreting results in this [video](https://www.youtube.com/watch?v=0v93qHDqq_g&feature=youtu.be&t=1h7m34s&source=post_page---------------------------):

- `pdpbox` (`R` package `pdp`)
- `treeinterpreter`


## Individual Conditional Expectation (ICE) Plots

[paper](https://arxiv.org/abs/1309.6392)

This is similar to PDP. ICE measures that dependency of a model on $X_C$, whereas PDP measures the average marginal effect of $X_C$ for the model.

Instead of holding $X_C$ constant, ICE chooses and example $x^{(i)}$, fixes $x^{(i)}_S$ constant and iterates through all possible values of $X_C$.

## Stratified Partial Dependence (StratPD)

[paper](https://arxiv.org/abs/1907.06698)

Both ICE and PDP rely on models. StratPD is model-independent. Because they both
interate through the data for all possible values of $X_C$, sometimes
illogic new samples are created, e.g. a 1 bedroom flat with 3 bathrooms.

Neither method has a way of showing that the model may not have seen some
areas of the space through data. I.e. 15 bedroom house prices, or pregnant males...

PDP/ICE also assumes that features are independent. Co-dependces, aka colinearity, causes problem for PDP/ICE interpretations.

### Paper Notation

$\mathrm{X}$ - data matrix $n \times p$ with $p$ features.

$y$ - $n \times 1$ reponse

$X$ - randomly selected example from $\mathrm{X}$

$y = f(X)$ - model

$C$ - index of features of interest. $\overline{C}$ - complement of $C$.

$x_c$ - a single feature, column of $\mathrm{X}$

### Stratification

1. Stratify $\mathrm{X}$ into **disjoint collections** of observations for which $x_{\bar{c}}$ are **constant** within each collection (ignoring $x_c$). The **goal** here is to find groups of extremely similar $x_{\bar{c}}$ values in $(X, y)$ so fluctuations in $y$ are due solely to changes in $x_c$.
   - $G_j$ be the index set of $j$th such collection of observations.
   - $\{(x_{i,c}, y_i)\}, i \in G_j$ describes how $x_c$ affects $y$ (because $x_{\bar{c}}$ is held constant here)

2. Fit linear regression model:

$$ y_i = \beta_{0, j} + \hat{\beta}_{G_j} x_{i, c} + \epsilon_i, i \in G_j$$

3. Partition the domain of $x_c$ into **disjoint regions**, $\{R_1, \cdots, R_m\}$, and let $\mathcal{I} = \{ G_j : G_j \cap \ R \neq \varnothing \}$ where $R \in \{R_1, \cdots, R_m\}$.

4. Partial dependence between $x_c$ and $y$ in a region $R$ is estimated as follows, i.e. it's the **average** of $\hat{\beta}_{G_i}$ in $\mathcal{I}$, **weighted** by the number of samples in the each collection in $\mathcal{I}$:

$$ \hat{\beta}_R = \frac{1}{\sum_{G_i \in \mathcal{I} \mid G_j \mid}} \sum_{G_j \in \mathcal{I}} \mid G_j \mid \hat{\beta}_{G_j} $$

Conceptually, we first stratify by **features not of interest** into collections, fit OLS model to obtain $\hat{\beta}_{G_i}$. The stratify differently by $x_c$, the **features of interest** into **regions**. For **each region**, we compute PD. How?
Find the collections that overlap with this region in data sample space, compute the **weighted average** of $\hat{\beta}_{G_i}$.

**My thoughts**: since the core idea of stratification is to find $x_{\bar{c}}$ that are similar, how about using clustering algos instead of a decision tree to find the collections of $G_i$? The benefit of using a decision tree in a supervised way is to link $X_{\overline{C}}$ with the response $y$. In an unsupervised setting we would lose that.

Decision trees choose feature space hypervolues that **minimise the variance in $y$**.

### Experiments

Section 5.2 illustrates the impact of model choice on the result of PDP/ICE. Many non-linear models produced PDP/ICE charts that failed to identify the true linear relationship betwee $X_i$ and $y$.

`PDP/ICE` does not work well when there are duplicated features. Some model would suffer from this. `StratPD` using `RF` can compensate this problem.

**My thoughts**: some of the advantages the authors pointed out would only manifest when one has knowledge of the problem, e.g. knowing there are **duplicated** features and therefore use `RF` and using `min_split_fetures=1` intead of a single tree for stratification; or changing the default `min_samples_leaf=` parameter from 10 to 2 based on knowledge of the dataset distribution (co-linearity, section 5.1). These issues may not be known to someone trying to use `StratPD` from the start, so one can easily fall into the same traps as `PDP/ICE` unless they tried running with different parameters, or specifically look for such issues prior to interpreting dependency. These is some signs of tuning `StratPD` to show favouriable results. On the upside it does illustrate that `StratPD` have settings that one can tune to cope with these issues.

**Pathological partitioning** can happen when:

1. $x_{\bar{c}}$ contains a singla categorical variable, or
2. when the only strongly-predictive variable in $x_{\bar{c}}$ is categorical.

When these happen, a single partition could contain a large number of observations.

### Noise in Data

The effect of **Noisy features** is limited for `StratPD` because decision trees ignore variables with low predictive power (in the process of forming collections $G_j$). For `PDP/ICE` if the underlying model is tree-based then this effect is also reduced. However, if other models are used, `PDP/ICE` may be more prone to noisy features.

**Noise in the response $y$** may require tuning `min_samples_leaf=` parameter.

`StratPD` seems to be more sensitive to noisy variables than `PDP/ICE`.

### Hyperparameter Tuning

Use `stratx.plot_stratpd_gridsearch()` to tune `min_samples_leave=` parameter.

For **categorical** features, having a `min_samples_leaf=` set to a value less than the no. of categories can cause inaccurate estimates. See section 5.7 in paper for example.

Generally speaking, large `min_samples_leaf=` is desirable because they are able to capture more non-linearities and is less susceptible to noise.

## TODO:

https://stats.stackexchange.com/questions/50560/how-to-calculate-partial-dependence-when-i-have-4-predictors

https://www.alexpghayes.com/blog/understanding-multinomial-regression-with-partial-dependence-plots/

http://rstudio-pubs-static.s3.amazonaws.com/283647_c3ab1ccee95a403ebe3d276599a85ab8.html

https://medium.com/@hiromi_suenaga/machine-learning-1-lesson-4-a536f333b20d
