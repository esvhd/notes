# `cvxportfolio`

Notes based on `cvxportfolio` paper on multi-period optimization.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

## Notation

$t$ - discrete time periods.

$h_t \in \mathbb{R}^{n+1}$ - denotes the portfolio (or vector) of **dollor holdings** at time period $t$.

$(h_t)_i$ - denotes the dollar holding at time period $t$ for asset $i$, for $i = 1, \cdots, n$, where $n$ is the no. of assets.
A positive number means long, negative number means short.

$(h_t)_{n+1}$ denotes cash position. 0 value means portfolio is fully invested.

$p_t \in \mathbb{R}^n_{+}$ - asset **prices** for time period $t$. Only non-negative values allowed.

$v_t$ - total value of the portfolio at time $t$, in dollars. $v_t = \mathcal{1}^T h_t$.

$\Vert (h_t)_{1:n} \Vert_{1}$ - denotes gross exposure, i.e. sum of absolute values of all positions.

$w_t \in \mathbb{R}^{n+1}$ - position weights, i.e. $h_t / v_t$. By definition weights sum up to 1.

$\Vert w_{1:n} \Vert$ - L1-norm of the portfolio, aka, portfolio **leverage**.

$u_t \in \mathbb{R}^{n}$ denote the **dollar value of trades**. Positive value indicates buy, negative means sell.
By same convention, $(u_t)_{n+1}$ denotes cash change.

$z_t = u_t / v_t$ denotes **trade weights**.

$h^{+}_t = h_t + u_t$ denotes **post-trade portfolio**.

**Post-trade wegith**: $w^+_t = w_t + z_t$

**Total change of portfolio value**: $v^{+}_t - v_t = 1^T h^{+}_t - 1^T h_t = 1^T u_t$

**Turnover** is defined as **half** of L1-norm: $\Vert (u_t)_{1:n} \Vert _1 / 2$. In percentage terms: $\Vert (z_t)_{1:n} \Vert _1 / 2$.

**Transaction Cost**: $\phi^{trade}_t(u_t)$, where $\phi^{trade}_t : \mathbb{R}^{n+1} \rightarrow \mathbb{R}$ is the **dollar transaction cost function**.

**Assumption**:

- $\phi^{trade}_t$ does **not** depend on $(u_t)_{n+1}$, i.e. no t-cost for cash position.

- $\phi^{trade}_t(0) = 0$, i.e. no t-cost when there is no trade.

- $\phi^{trade}_t$ is **separable**, i.e. $\phi^{trade}_t(x) = \sum^{n}_{i=1} (\phi^{trade}_t)_i (x_i)$
