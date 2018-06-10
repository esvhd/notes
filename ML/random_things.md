# Notes on random stuff I read about...

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


