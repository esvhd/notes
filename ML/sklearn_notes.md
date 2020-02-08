# `sklearn` Notes

## Sources

- Hands on Machine Learning with sklearn, Keras & Tensorflow - HOML

## Notation

- `m` - # of samples
- `n` - # of features

## SVM

`LinearSVC(loss='hinge')` is faster than `SVC(kernel='linear')`, especially
for larger data. See HOML p162 for discusson between the two.

Complexity:


- `LinearSVC/liblinear` - $O(m \times n)$, use `LinearSVC(tol=)` to select required precision.
- `SVC/libsvm` - $O(m^2 \times n)$ or $O(m^3 \times n)$


With `LinearSVC`, set `loss='hinge'` as it's not the default.
Also set `dual=False`, unless there are more features than training examples.

**Kernel choices**: Rule of thumb is to always try the linear kernel first.
Then if dataset is not too large, try `rbf`.

`SGDClassificer(loss='hinge')` uses SGD to solve for SVM, works better for
large dataset that cannot fit into memory. When using SGD, remeber that it works
when data is IID, on average. Therefore, data should be shuffled so lables are
mixed. Otherwise, SGD will start optimizing for one label, then the next, etc.

SVMs can be used for outlier detection. See [here](https://scikit-learn.org/stable/modules/outlier_detection.html)

## Decision Trees

`sklearn` uses `CART` algo so always produces binary trees.

Complexity:

- prediction: $O(\log_2(m))$
- training: $O(n \times m \log_2 (m))$

Gini impurity or Entropy: Gini impurity is faster, most of the time they are
similar. When they differ, Gini impurity tends to isolat ethe most frequent
class in its own branch of the tree, while entropy tends to produce slightly
more balanced trees. HOML p181

Decision trees love orthogonal decision boundaries, therefore they are prone to
training set rotation. Running PCA prior to decision tree often limits this
issue.

