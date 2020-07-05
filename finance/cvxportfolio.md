<!-- MarkdownTOC levels="1,2,3", autolink=true -->
<!-- /MarkdownTOC -->

# `cvxportfolio`

Notes based on `cvxportfolio` paper on multi-period optimization.

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

**Post-trade wegith**: $w^+_t = w_t + z_t$

$h^{+}_t = h_t + u_t$ denotes **post-trade portfolio**.

**Returns**: $r_t \in \mathbb{R}^{n+1}$ is the vector of **assets and cash** geometric returns from period $t$ to period $t+1$.

We have: $h_{t+1} = (1 + r_t) \odot h^+_t$, i.e. next period $t+1$ portfolio is post-trade portfolio at time $t$ times return between time $t$ and $t+1$.

**Total change of portfolio value**: $v^{+}_t - v_t = 1^T h^{+}_t - 1^T h_t = 1^T u_t$

**Turnover** is defined as **half** of L1-norm: $\Vert (u_t)_{1:n} \Vert _1 / 2$. In percentage terms: $\Vert (z_t)_{1:n} \Vert _1 / 2$.

**Transaction Cost**: $\phi^{trade}_t(u_t)$, where $\phi^{trade}_t : \mathbb{R}^{n+1} \rightarrow \mathbb{R}$ is the **dollar transaction cost function**.

**Assumption**:

- $\phi^{trade}_t$ does **not** depend on $(u_t)_{n+1}$, i.e. no t-cost for cash position.

- $\phi^{trade}_t(0) = 0$, i.e. no t-cost when there is no trade.

- $\phi^{trade}_t$ is **separable**, i.e. $\phi^{trade}_t(x) = \sum^{n}_{i=1} (\phi^{trade}_t)_i (x_i)$


## Transaction Cost Model

**Scaler t-cost function**:

$$ x \rightarrow a \vert x \vert + b \sigma \frac{|x|^{3/2}}{V^{1/2}} + c x $$

Where:

- $x$ is a dollar trade amount
- $a \in \mathbb{R}^n$ is $1/2$ of bid-ask spread for asset at beginning of time period as a % of price
- $b \in \mathbb{R}^n$ is a positive constant with unit inverse dollars.
- $V \in \mathbb{R}^n$ is the total market volume traded for the asset in the time period, in dollar value.
- $\sigma \in \mathbb{R}^n$ - price volatility over recent time periods, in dollars.
- $c \in \mathbb{R}^n$ used to create asymmetry for buys and sells. when $c = 0$, buy = sell t-cost. When $c > 0$, sell is cheaper.
- $\frac{3}{2}$ power t-cost model is a widely used one by practitioners.

Thoughts: This is a model that works for equities or exchange traded products where volume is known. Not useful for OTC bond market.

## Holding Cost

Essentially funding cost for short positions, plus any negative carry.

$$ \phi^{hold}_{t}(h^{+}_t) = s^T_t (h^{+}_t)_{-} $$

Where:

- $s^T_t  \in \mathbb{R}^n \ge 0$ is the borrowing cost, in period $t$, for asset $i$
- $(z)_{-} = \max\{-z, 0 \}$ denotes the **negative** part of a number $z$.

To normalise by portfolio value, we devide by $v_t$:

$$ \phi^{hold}_{t}(h^{+}_t) / v_t = s^T_t (w_t + z_t)_{-} $$


## Returns

Next period portfolio value:

$$
\begin{aligned}
v_{t+1} &= 1^T h_{t+1} \\
&= (1 + r_t)^T h^+_t \\
&= (1 + r_t)^T (h_t + u_t) \\
&= (1 + r_t)^T h_t + (1 + r_t)^T u_t \\
&= 1^T h_t + r^T_t h_t + (1 + r_t)^T u_t \\
&= v_t + + r^T_t h_t + (1 + r_t)^T u_t \\
&= v_t + + r^T_t h_t + r^T_t u_t + 1^T u_t\\
&= v_t + + r^T_t h_t + r^T_t u_t - \phi^{trade}_t(u_t) - \phi^{hold}_t (h^+_t)
\end{aligned}
$$

Essentially, it's the sum of:

- value of portfolio at time $t$
- return of holdings at time $t$
- return of trades at time $t$
- cost of trades
- cost of holdings

**Portfolio Return** for time $t$ to $t+1$ defined as $R^P_t$:

$$
\begin{aligned}
R^P_t &= \frac{v_{t+1} - v_t}{v_t} \\
&= r^T_t w_t + r^T_t z_t - \phi^{trade}_t(z_t) - \phi^{hold}_t (w_t + z_t)
\end{aligned}
$$

The terms here are similar to the above, i.e. the last four bullet points.

**Next period weights** defined as:

$$ w_{t+1} = \frac{(1 + r_t) \odot (w_t + z_t)}{1 + R^P_t} $$

### Non-instance trading

The easiest way to deal with non-instance trade is to model returns with higher frequence, ie. instead of daily, use hourly.

Thought: this is not really easier, as the requirement for data is higher and for other markets like bonds, this is hard to achieve.

### Multi-period price impact

This is not modeled in this paper. Large order can impact asset prices in future periods.

## Active and Excess Return

**Active return** is given by:

$$ R^a_t = R^P_t - R^b_t $$

**Excess return** is used when the benchmark is a risk free rate:

$$ R^e_t = R^P_t - (r_t)_{n+1} $$

Define **Average Active Return** as:

$$
\begin{aligned}
R^a_t &= R^P_t - R^b_t \\
&= r^T_t (w_t - w^b_t) + r^T_t z_t - \phi^{trade}_t(z_t) - \phi^{hold}_t (w_t + z_t)
\end{aligned}
$$


## Single-Period Optimisation

### Notation

**Predicted returns**: $\hat{r}_t$ for time period $t$.

**Predicted portfolio return**:

$$ \hat{R}^P_t = \hat{r}^T_t w_t + \hat{r}^T_t z_t - \hat{\phi}^{trade}_t(z_t) - \hat{\phi}^{hold}_t (w_t + z_t) $$

### Optimization

Solve for trade $z_t$ by solving:

$$
\begin{aligned}
\max \;\; &\hat{R}^P_t - \gamma_t \psi_t (w_t + z_t) \\
\text{subject to} \;\; &z_t \in \mathcal{Z_t} \\
&w_t + z_t \in \mathcal{W_t} \\
&1^T z_t + \phi^{trade}_t(z_t) + \phi^{hold}_t (w_t + z_t) = 0
\end{aligned}
$$

Where:

- $\psi_t : \mathbb{R}^{n+1} \rightarrow \mathbb{R}$ is a risk function
- $\gamma_t > 0$ is the **risk aversion parameter**
- $\mathcal{Z_t}$ - trade constraints
- $\mathcal{W_t}$ - holding constriants

**Self-financing constraint**:

$$1^T z_t + \phi^{trade}_t(z_t) + \phi^{hold}_t (w_t + z_t) = 0$$

This essentially says the trades plus t-cost and funding costs should not impact cash position. We can **simplify** this constraint to just:

$$ 1^T z_t = 0 $$

I.e. ignore t-cost and funding costs, which in practise tend to be small. By doing so, the optimization problem slightly **over-estimates** post-trade cash balance (by ingoring t-cost / funding cost in constraint).

For easier implementation, we can rewrite the problem to with $w_{t+1}$:

$$
\begin{aligned}
\max \;\; &r^T_t w_{t+1} - \phi^{trade}_t(w_{t+1} - w_t) - \phi^{hold}_t (w_{t+1}) - \gamma_t \psi_t (w_{t+1}) \\
\text{subject to} \;\; &w_{t+1} - w_t \in \mathcal{Z_t} \\
&w_{t+1} \in \mathcal{W_t} \\
&1^T w_{t+1} = 1
\end{aligned}
$$

### Risk Function $\psi_t$

This risk function can take many forms:

- Absolute risk
- Active risk
- Factor risk model
- Transformed risk
- Worst-case quadratic risk

#### Transformed risk

Think of this as a way to automatically tune the risk aversion parameter, increasing it as risk increases.

We can make $\psi_t(x)$ a non-linear monotonically increasing function for example.

Other examples:

- $\psi_t(x) = (x - a)_+$, i.e. no cost if risk level below $a$, then linear.
- $\psi_t(x) = \exp(x / \eta), \;\eta > 0$ - exponential cost when x is above $\eta$

#### Worst-case quadratic risk

This is an interesting case, where the risk function is defined as the **max** of $M$ given scenarios / covariance matrices.

$$ \psi_t(x) = \max_{i=1,\cdots,M} (x - w^b_t)^T \Sigma^{(i)}_t (x - w^b_t) $$

Where $\Sigma^{(i)}_t$ is the covariance matrix for scenario $i$.

### Forecast Error Risk
