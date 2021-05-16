# Learning to Rank

Notes on Burges's paper From RankNet to LambdaRank to LambdaMART.

## Notation

The problem is framed as given a query, how do we rank the results returned.

- $U_i, U_j$ - URLs $i$ and $j$ returned by a query, with feature vectors $x_i$, $x_j$

- $U_i \vartriangleright U_j$ - $U_i$ ranks higher than $U_j$

- $s_i, s_j$ - Scores from a function $f(x_i)$ and $f(x_j)$

- $S_{ij} \in \{ -1, 0, +1\}$ - labels for pair $ij$, -1 = $i$ ranks lower, 0 = same rank, +1 = $i$ ranks higher.

- $P_{ij} = P(U_i \vartriangleright U_j)$ - probability of $U_i$ ranking higher than $U_j$. defined as:

$$ P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$$

Where $\sigma()$ is the sigmoid function.

- $\bar{P}_{ij}$ - indicates ground truth labels for $U_i$ ranking higher than $U_j$.
  We can express the below relationships: $\bar{P}_{ij} = \frac{1}{2} (1 + S_{ij})$
  - When $S_{ij} = -1$, $\bar{P}_{ij} = 0$
  - When $S_{ij} = 0$, $\bar{P}_{ij} = 0.5$
  - When $S_{ij} = 0$, $\bar{P}_{ij} = 1$

## RankNet

Some useful terms for expansion later:

$$
\begin{aligned}
z &= -\sigma(s_i - s_j) \\
\log{P_{ij}} &= \log\frac{1}{1 + e^z} = \log(1) - log(1 + e^z) = - \log(1 + e^z) \\
\log({1 - P_{ij}}) &= \log\bigg[1 - \frac{1}{1 + e^z} \bigg] \\
&= \log \bigg[ \frac{1 + e^z - 1}{1 + e^z}\bigg] \\
&= \log(e^z) - \log(1 + e^z)
\end{aligned}
$$

Using gradient descent to solve a ranking problem with cross entropy loss below,
and expanding with other terms:

$$
\begin{aligned}
C &= -\bar{P}_{ij} \log (P_{ij}) - (1 - \bar{P}_{ij}) \log(1 - P_{ij}) \\
&= -\frac{1}{2} (1 + S_{ij}) \log(P_{ij}) - \bigg[1 - \frac{1}{2} (1 + S_{ij})\bigg] \log(1 - P_{ij}) \\
&= -\frac{1}{2} (1 + S_{ij}) (- \log(1 + e^z)) - \bigg[1 - \frac{1}{2} (1 + S_{ij})\bigg] \log(1 - P_{ij}) \\
&= \frac{1}{2} (1 + S_{ij}) \log(1 + e^z) - \bigg[1 - \frac{1}{2} - \frac{S_{ij}}{2}\bigg] \log(1 - P_{ij}) \\
&= \frac{1}{2} (1 + S_{ij}) \log(1 + e^z) - \bigg[\frac{1}{2} - \frac{S_{ij}}{2}\bigg] \log(1 - P_{ij}) \\
&= \frac{1}{2} (1 + S_{ij}) \log(1 + e^z) - \frac{1}{2} \big(1 - S_{ij} \big) \big[\log(e^z) - \log(1 + e^z) \big] \\
&= \frac{1}{2} (1 + S_{ij}) \log(1 + e^z) - \frac{1}{2} \big(1 - S_{ij} \big) \log(e^z) + \frac{1}{2} \big(1 - S_{ij} \big) \log(1 + e^z) \\
&= \log(1 + e^z) \bigg[\frac{1}{2} (1 + S_{ij}) + \frac{1}{2} \big(1 - S_{ij} \big)\bigg]- \frac{1}{2} \big(1 - S_{ij} \big) \log(e^z) \\
&= \log(1 + e^z) - \frac{1}{2} \big(1 - S_{ij} \big) \log(e^z) \\
&= \log(1 + e^z) - \frac{1}{2} \big(1 - S_{ij} \big) z \\
\end{aligned}
$$

This is the first equation on page 3 of the paper. Note that:

$$ \frac{\partial{C}}{\partial{s_i}} = -\frac{\partial{C}}{\partial{s_j}} $$

## Infomation Retriveal Measures

**MRR** = Mean Reciprocal Rank, binary relevance

**MAP** = Mean Average Precision, binary relevance

**NDCG** = Normalised Discounted Cumulative Gain. NDCG $\in [0, 1]$,
evaluated for each query group, and averaged over all groups.

$$ DCG @ T = \sum_{i=1}^T \frac{2^{l_i}-1}{\log_2(1 + i)}  $$

Where:

- $T$ is the truncation level, e.g. $T=10$ if we care about the first 10 results
- $i$ ranking position of predicted results, $i \in \{0, 1, 2, \cdots\, T\}$
- $l_i$ is the label of the $i$th listed URL.

$$ NDCG = \frac{DCG @ T}{\max DCG @ T} $$

Where $\max DCT @ T$ is defined as the best DCG possible, i.e. the score achieved
by perfect prediction. To comupte this, we simply sort by true label and compute
DCG.

Thanks to this [answer](https://stats.stackexchange.com/questions/303385/how-does-xgboost-lightgbm-evaluate-ndcg-metric-for-ranking), `LightGBM`
assigns missing queries (a query with no returned result)
with $\max DCT @ T=1$.

NDCG Limitations:
[here](https://www.geeksforgeeks.org/normalized-discounted-cumulative-gain-multilabel-ranking-metrics-ml/)

- NDCG does not penalise bad outputs, e.g. returning only relevant docs is not
  better than returning relevant docs PLUS a number of irrelevant docs.
- Does not deal with missing any relevant documents in outputs, e.g. returning
  [3, 3, 3] vs [3, 3, 3, 3] produce equal NDCG score.

When a query contains documents **not** included in the true labels, in computing
DCG, its associated score is assumed to be 0. See `rank_eval.metric._dcg` code
for this, [link](https://github.com/AmenRa/rank_eval/blob/master/rank_eval/metrics.py#L132-L140). Note need to uncomment `@njit` for `_dcg()` to be able to print
the values. This clearly **assumes that 0 is the worst relevance score you
can have**.

Also with this orignial definition of DCG, one can run into numerical problems
when the returned results are large in nubmers. Think 1k, e.g. $2^{1000} - 1$.
Therefore, an alternative implementation uses:

$$ DCG @ T = \sum_{i=1}^T \frac{l_i}{\log_2(1 + i)}  $$

Some open source libs for ranking evaluation:

- [`rank_eval`](https://github.com/AmenRa/rank_eval)
- [`rankeval`](https://github.com/hpclab/rankeval)

**ERR** = Expected Reciprocal Rank

$$ ERR = \sum_{r=1}^{n} \frac{1}{r} R_r \prod_{i=1}^{r-1} (1 - R_i) $$

Where:

- $l_m$ is the maximum label value
- $R_i = \frac{2^{l_i} - 1}{2^{l_m}}$ - the probability that a user finds the
  document at rank position $i$ relevant.

NDCG & ERR can handle multiple levels of relevance.
