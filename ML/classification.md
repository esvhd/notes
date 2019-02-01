# Classification Notes

<!-- MarkdownTOC levels="1,2,3,4" autolink=true -->

- [Imbalanced Classification Problems](#imbalanced-classification-problems)
    - [Experiment Construction](#experiment-construction)
    - [Findings](#findings)
    - [Data Level Methods](#data-level-methods)
        - [Oversampling](#oversampling)
        - [Undersampling](#undersampling)
    - [Classifier Level Methods](#classifier-level-methods)
        - [Thresholding](#thresholding)
        - [Cost Sensitive Training](#cost-sensitive-training)
        - [One vs Other](#one-vs-other)
    - [Hybrid of Methods](#hybrid-of-methods)

<!-- /MarkdownTOC -->


## Imbalanced Classification Problems

Some notes based on this [paper](https://arxiv.org/abs/1710.05381)

### Experiment Construction

The paper tests two broad types of class imbalance problems.

- **Step** imbalance, where the minority classes were similarly imbalanced (had simliar number of samples) vs the majority classes which also had simliar number of samples.
- **Linear** imbalance, from the least minority class to the most majority class, the number of samples linearly increases.

### Findings

Measured by **multi-class ROC AUC**:

* Oversampling dominated in almost all analyzed scenarios.
* Oversampling should be applied to the level that completely eliminates the imbalance.
* Oversampling does **not** cause overfitting of CNNs, unlike classical ML methods.
* Thresholding should be applied to compensate for prior class probabilities when overall number of properly classified cases is of interest.

### Data Level Methods

#### Oversampling

Some methods used for oversampling:

* Basic oversampling
* SMOTE - augments with artificial data created by interpolating neighbouring data points.
* Extension of SMOTE, e.g. to focus on examples near the boundary between classes.
* Cluster based, first clusters data, then sample from each cluster. Reduces both between-class and within-class imbalance.
* DataBoost-IM, identify difficult examples with boosting preprocessing and use them to generate synthetic data.
* Neural nets - ensure uniform class distribution in each minibatch and control the selection of examples from each class.

#### Undersampling

Loses a lot of data... Big disadvantage.

For extreme ratios of imbalance and a large portion of classes are minorities, undersampling can perform simliar to oversampling. In these cases, if training time is a concern, undersampling may be preferred.


### Classifier Level Methods

#### Thresholding

This [paper](http://www.ee.iisc.ac.in/people/faculty/prasantg/downloads/NeuralNetworksPosteriors_Lippmann1991.pdf) is interesting. It shows that neural nets directly estimates Bayesian **a posterior** probabilities. Therefore, estimating:

$$ p(C_i \mid x) = \frac{p(x \mid C_i) \times p(C_i)}{p(x)} $$

To compensate for varying **a prior** probabilities (see paper section 4.1), one can divide the network output by $p(C_i)$, and then mutiply by a different prior, $p'(C_i)$, based on a **test set** for inference. 

$p(C_i)$ can be estimated by counting the number of class $C_i$ in the **training data**. 

$p'(C_i)$ is estimated by counting the number of class $C_i$ in the **test set**.

#### Cost Sensitive Training

Paper mentions a few methods:

* threshold moving
* post scaling
* modify the learning rate for hard examples
* Use mis-classification cost instead of cross entrpy as the loss function. This is equivalent to oversampling. I thought this was quite interesting.

#### One vs Other 

Turns the classification problem into a **anomaly detection** problem.

### Hybrid of Methods

Use a mix of methods from methods described above. Or use ensembling. 
