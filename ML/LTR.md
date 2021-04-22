# Learning to Rank

Notes on Burges's paper From RankNet to LambdaRank to LambdaMART.

## Notation

The problem is framed as given a query, how do we rank the results returned.

- $U_i, U_j$ - URLs $i$ and $j$ returned by a query, with feature vectors $x_i$, $x_j$

- $U_i \vartriangleright U_j$ - $U_i$ ranks higher than $U_j$

- $s_i, s_j$ - Scores from a function $f(x_i)$ and $f(x_j)$

- $S_{ij} \in \{ -1, 0, +1\} $ - labels for pair $ij$, -1 = $i$ ranks lower, 0 = same rank, +1 = $i$ ranks higher.

- $ P_{ij} = P(U_i \vartriangleright U_j) $ - probability of $U_i$ ranking higher than $U_j$. defined as:

$$ P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$$

Where $\sigma()$ is the sigmoid function.

- $ \bar{P}_{ij} $ - indicates ground truth labels for $U_i$ ranking higher than $U_j$.
  We can express the below relationships: $ \bar{P}_{ij} = \frac{1}{2} (1 + S_{ij}) $
  - When $S_{ij} = -1$, $ \bar{P}_{ij} = 0 $
  - When $S_{ij} = 0$, $ \bar{P}_{ij} = 0.5 $
  - When $S_{ij} = 0$, $ \bar{P}_{ij} = 1 $

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

**ERR** = Expected Reciprocal Rank

**NDCG** = Normalised Discounted Cumulative Gain. NDCG $\in [0, 1]$

NDCG & ERR can handle multiple levels of relevance.

$$ DCG @ T = \sum_{i=1}^T \frac{2^{l_i}-1}{\log(1 + i)}  $$

Where:

- $T$ is the truncation level, e.g. $T=10$ if we care about the first 10 results
- $l_i$ is the label of the $i$th listed URL.

$$ NDCG = \frac{DCG @ T}{\max DCG @ T} $$

$$ ERR = \sum_{r=1}^{n} \frac{1}{r} R_r \prod_{i=1}^{r-1} (1 - R_i) $$

Where:

- $l_m$ is the maximum label value
- $R_i = \frac{2^{l_i} - 1}{2^{l_m}}$ - the probability that a user finds the
  document at rank position $i$ relevant.
