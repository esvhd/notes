# Notes on Feature Importance and Interpretation

<!-- MarkdownTOC levels='1,2,3' autolink="true" -->

- [Notes on Feature Importance and Interpretation](#notes-on-feature-importance-and-interpretation)
  - [Logistic Regression](#logistic-regression)
  - [Permutation Importance for Random Forest Feature Importance](#permutation-importance-for-random-forest-feature-importance)
    - [Additional Notes](#additional-notes)
  - [Partial Dependency Plots (PDP)](#partial-dependency-plots-pdp)
    - [Drawbacks for PDP](#drawbacks-for-pdp)
  - [Individual Conditional Expectation (ICE) Plots](#individual-conditional-expectation-ice-plots)
  - [ALE (Accumulated Local Effects)](#ale-accumulated-local-effects)
    - [Comparison](#comparison)
  - [Drawback of Permutation Importance](#drawback-of-permutation-importance)
  - [Shapley Adaptive Explanation (`SHAP`)](#shapley-adaptive-explanation-shap)
    - [Notation](#notation)
    - [Additive Feature Attribution Methods](#additive-feature-attribution-methods)
    - [LIME](#lime)
    - [Shapley Value](#shapley-value)
    - [Classic Shapley Value Esitmation](#classic-shapley-value-esitmation)
    - [SHAP](#shap)
  - [Stratified Partial Dependence (StratPD)](#stratified-partial-dependence-stratpd)
    - [Paper Notation](#paper-notation)
    - [Stratification](#stratification)
    - [Experiments](#experiments)
    - [Noise in Data](#noise-in-data)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [TODO:](#todo)

<!-- /MarkdownTOC -->

## Logistic Regression

$$ \log\big(\frac{p(y=1)}{1 - p(y=1)} \big) = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p $$

A change in a feature $i$ by one unit changes the **odds ratio** by a factor of $e^{\beta_i}$.

Edge case: logistic regression can suffer from **complete separation**: if there is a feature that completely separates the data, logistic regression can no longer be trained - the weight for that feature would **not** converge.


## Permutation Importance for Random Forest Feature Importance

See this [post](http://parrt.cs.usfca.edu/doc/rf-importance/index.html), [github](https://github.com/parrt/random-forest-importances)

`sklearn` Random Forest feature importance and `R`'s default Randome Forest feature importance strategies based on **minimum decrease in impurity** are **biased**,
i.e. it tends to inflate the importance of continuous or high-cardinality
categorical variables. Example in the articule showed an example with a random
value feature being ranked more important than actual features, with regression
tree.

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

In 2019 a [paper](https://arxiv.org/pdf/1905.03151.pdf) published argued for
not using permutation importance - because they favour correlated features.
This issue was also highlighted by Marcos Lopez de Prado in his books.
Marcos also proposed a method to permute features based on correlation clusters.

Other alternavies include:

- Drop column and retrain, measure performance differences.
- Permutate data and retrain,
- Conditionally generate / permute features conditioned on other features.
  Similar to permuting groups of features
- SHAP

### Additional Notes

Lopez de prado pointed out some very practical considerations when using permutation importance in ML for AM book.

**Subsititution Effets**: when two or multiple features share predictive info. Two identical features may be considered
both unimportance in a Mean Decrese in Accuracy (**MDA**) metric such as permutation importance.

**Co-dependence** of features results in subsititution effects, one solution to multicollinearity is to apply PCA
on features first, and then use MDA.

**Better** approach is to cluster similar features and apply feature importance analysis at the cluster level.

**Silhouette coefficient**:

Effectively this is a measure comparing intra-cluster distance and inter-cluster distance.

$S_i = \frac{b_i - a_i}{\max\{a_i, b_i\}} \forall i = 1, \cdots, N$

Where:

- $a_i$ is the average distance between $i$ and all other elements in the same cluster, and

- $b_i$ is the average distance between $i$ and all elements in the nearest cluster of which $i$ is not a member

$S_i = 1$ means $i$ is clutered well, $S_i = -1$ means $i$ is clustered poorly.

## Partial Dependency Plots (PDP)

Based on **ESL** p369-370.

Definitions: let feature space be $X \in \mathbb{R}^{N \times p}$, let $X_S \in \mathbb{R}^{N \times l}$ where $l < p$ be a subset of features. Let $C$ be the **complement** set, so $S \cup C = \{1, 2, \cdots, p\}$.

One way to define **average** or **partial** dependence of $f(X)$ on $X_S$ is:

$$ f_S (X_S) = \mathbb{E}_{X_C} f(X_S, X_C) $$

What this means is that we compute the following:

$$ \bar{f}_S (X_S) = \frac{1}{N} \sum_{i=1}^{N} f(X_S, x^{(i)}_C) $$

where $x^{(i)}_C$ is the $i$th example out of $N$ observations in the complement $C$ set, i.e. use $X_S$ and $x^{(i)}_C$ to form a new dataset (for features in $C$ we use values of $x^{(i)}_C$), then pass through the model, repeat $N$ times (using each of the $N$ examples), then take the average.

In other words, for each unique value of $X_S$, create a new dataset with $X_C$ (i.e. N passes), run prediction and take average.
Complexity = $nunique(X_S) \times N$. Realistically, some sampling is done here instead of using unique values of $X_S$.

This is computationally intensive. Fortunately, for trees, this can be done efficiently without reference to the data.

Jeremy Howard showed some interesting python packages for interpreting results in this [video](https://www.youtube.com/watch?v=0v93qHDqq_g&feature=youtu.be&t=1h7m34s&source=post_page---------------------------):

- `pdpbox` (`R` package `pdp`)
- `treeinterpreter`
- `Skater` - python
- `R`: `iml` or `DALEX`

### Drawbacks for PDP

- Some regions of plots are not realistic when features are correlated.
- **Heterogeneous effects**: when a feature has non-linear effects, e.g. half of larger value results in larger prediction,
  but for the other half smaller values results in larger prediction. PDP could show a straight line because the effect
  cancels each other on average, and the feature can be interpreted as having no effect on prediction. Solution?
  use **ICE** plots.

## Individual Conditional Expectation (ICE) Plots

[paper](https://arxiv.org/abs/1309.6392)

This is similar to PDP. ICE measures that dependency of a model on $X_C$, whereas PDP measures the average marginal effect of $X_C$ for the model.

Instead of holding $X_C$ constant, ICE chooses an example $x^{(i)}$, fixes $x^{(i)}_S$ constant and iterates through all possible values of $X_C$. I.e. for each data sample, hold $X_C$, compute for all values of $X_S$.

The average of ICE lines for $x_S$ corresponds to the PDP line for $x_S$.

Packages:

- `R`: `ml`, `ICEBox`, `pdp`
- Python: `condvis`

## ALE (Accumulated Local Effects)

To make up for the fact that some regions of PDP plots may not be realistic due to correlation between features,
ALE uses a nice trick to isoluate the effects on prediction for local regions of a feature.

M-plots (marginal plots) - for a given feature $S$ and value $V_S$, compute predictions for all samples with value $V_S$
(or similiar), take the average. The problem with M-plots is that then there are other features which are correlated
with $S$, this correlation would not be discovered - because when we iterate different values of $S$ and finding
samples for each value, these correlation cannot be isolated.

What do we do then? ALE uses a nice trick to block the effects of other features. Steps:

1. Divide the value range for feature $S$ into intervals with equal # of samples.
2. For each interval $i$, find samples that below to this internal.
3. For each sample $k$, compute $d_{i, k} = f_k(V_{i, max}) - f_k(V_{i, max})$
4. Compute average of $d_{i} = \frac{1}{N} \sum_{k=1}^{N} d_{i, k}$

**By taking the difference, the effect of other features are blocked.**

When deciding on the interval for a features, `qcut` is used to make sure all intervals have the same number of data
samples. The disadvantage here is that the intervals can have different length.

ALE packages:

- `R`: `iml` and `ALEPlot`
- `python`: `alibi`

### Comparison

| Method | No. features | Handle Correlated Features | Reveal Heterogeneous effects |
| --- |:---:|:---:|:---:|
| PDP | Max 2| N | N |
| ICE | Max 1| N | Y |
| ALE | | | |
| Shap | | | |

## Drawback of Permutation Importance

Methods such as permutation importance can be misleading when features are
correlated. The intuition here is that, if `x1` and `x2` are correlated, even
when `x1` is permuted, the model may still do ok with `x2` intact.

[Don't Permute-and-Predict](https://arxiv.org/abs/1905.03151) discusses the
effects here.

Lopez de Prado also proposed using clustering to find correlated features,
then permute the group as a whole.

## Shapley Adaptive Explanation (`SHAP`)

Python package [`shap`](https://github.com/slundberg/shap), [paper](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

[`treeexplainer` paper](https://arxiv.org/abs/1905.04610)

Kaggle has a short [course](https://www.kaggle.com/learn/machine-learning-explainability) on `SHAP`.

[Blog](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) from one of the creators.

Unifies a few different approaches:

- LIME
- DeepLIFT
- Layer-wise relevance propagation
- Shape Regression Values / Shapley Sampling Values
- Quantitative Input Influence

### Notation

- $f$ - original prediction model
- $x$ - single sample
- $f(x)$ - single prediction
- $x'$ - **simplified** input, through **mapping function** $x = h_x(x')$. In thi spaper $x'$ is a indicator vector to identify which features are present, i.e. $x' = (0, 1, 1)$ means feature $x0$ is not present but ($x_1, $_2)$ are.
- Local methods try to ensure $z' \approx x'$, and $g(z') \approx f(h_x(z'))$

### Additive Feature Attribution Methods

General form:

$$ g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i $$

where:

- $z' \in {0, 1}^M$
- $M$ is the number of simplified input features
- $\phi_i \in \mathbb{R}$

### LIME

[paper](https://arxiv.org/abs/1602.04938)

TODO: summarise paper

Cody's blog: LIME fits a simple linear model at the local level where the target $y$ is the prediction / output of the complex model. If the features are in the same scale, then the coefficients of this local linear model represents the importance of the features.

In LIME, **localness* is defined by a kernel distance function. I.e. draw samples from training dataset, then weight samples by the kernel function output.

### Shapley Value

Some notes based on [this](https://christophm.github.io/interpretable-ml-book/shapley.html)

For data matrix $X \in R^{n \times p}$, and model $f$, the Shapley value for feature $j$ in sample $x^{(i)}$, a.k.a $x^{(i)}_j$, is the contribution of feature $j$ to the prediction of $f(x^{(j)})$, compared to the **average prediction** for the data set.

Good blog [post](https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d) from Cody Marie Wild ()

- If a model is **linear**, or if the features are truly **independent**, then no matter what **values of the features** or the **sequence in which features are added to the model** , the contribution of a given feature is the **same**.

The Shapley value formula gives us a way to calculate the average impact of all permutations of how a feature could have contributed to the prediction.

In the `SHAP` paper, when predicting with $S\cup F \setminus {i}$ features actually means using the **average value of feature $i$** for training the dataset in the same model, i.e. **there isn't a second model that is trained without feature $i$**.

### Classic Shapley Value Esitmation

**Shapley regression values** is a feature importance metrics for linear regression models in the presence of multi-collinearity.

Define a feature set $F$, where $S$ is a subset of $F$, i.e. $S \subseteq F$. For a feature $i$, the Shapley regression value is computed as follows:

1. Fit a model **with** this feature, $f_{S\cup \{i\}}$, i.e. subset $S$ plus feature $i$
2. Fit a model **without** this feature, $f_S$. In `shap`'s implementation, this means usin gthe same model as step 1, but with feature $i$ replaced with its average value.
3. Prediction from both models are compared on the same inputs, $f_{S\cup \{i\}}(x_{S\cup \{i\}}) - f_S (x_S)$
4. This difference is computed for all possible subsets $S \subseteq F \setminus \{i\}$
5. Shapley regression value is the weighted average of these possible differences:

$$ \phi_{i} = \sum_{S \subseteq F \setminus \{i\}} \frac{\mid S \mid! (\mid F \mid - \mid S \mid - 1)!}{\mid F \mid} \big[ f_{S\cup \{i\}}(x_{S\cup \{i\}}) - f_S (x_S)\big]$$

Intuitively, $\mid S \mid!$ is the no. of permutations that $S$ can appear bfore $i$, $(\mid F \mid - \mid S \mid - 1)!$ is the no. of permutations that the remaining features can appear after $i$.

This is clearly computationally hard to do in practice. **Shapley sampling values** are approximations of this equation by **integrating over samples from the training set** (What does this mean?). This eliminates the need to retrain the model and allows fewer than $2^{\mid F\mid}$ differences to be computed.

Shapley regression value is the only attribution method that satisties the properties of **Efficient, Symmetry, Dummy and Additivity**.

### SHAP

TODO: summarise

SHAP values are the Shapley values of a conditional expectation function of the original model. They are solutions to:


$$ \phi_i(f, x) = \sum_{z' \subseteq x'} \frac{\mid z' \mid! (\mid M \mid - \mid z' \mid - 1)!}{\mid M \mid} \big[ f_{x}(z') - f_x (z' \setminus i)\big]$$

Where:

- $'z' \approx x'$, with mapping function $x = h_x(x')$
- $z' \setminus i$ denotes setting $z'_i = 0$
- $\mid z' \mid$ is the number of non-zero entries in $z'$
- $z' \subseteq x'$ represents all $z'$ vectors where the non-zero entries are a subset of the non-zero entries in $x'$
- $S$ is the set of non-zero indexes in $z$
- $h_x(z') = z_S$, where $z_S$ has **missing** values for features not in the set $S$
- Most models cannot handle **missing** values, we **approximate** $f(z_S) \approx E[f(z) | z_S]$
- $f_x(z') = f(h_x(z')) = E[f(z) | z_S]$


## Stratified Partial Dependence (StratPD)

[paper](https://arxiv.org/abs/1907.06698)

Both ICE and PDP rely on models. StratPD is model-independent. Because they both
interate through the data for all possible values of $X_C$, sometimes
illogic new samples are created, e.g. a 1 bedroom flat with 3 bathrooms.

Neither method has a way of showing that the model may not have seen some
areas of the space through data. I.e. 15 bedroom house prices, or pregnant males...

PDP/ICE also assumes that features are independent. Co-dependces, aka collinearity, causes problem for PDP/ICE interpretations.

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
