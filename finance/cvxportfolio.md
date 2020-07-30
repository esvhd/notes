# `cvxportfolio`

<!-- MarkdownTOC -->

- Notation
- Transaction Cost Model
- Holding Cost
- Returns
    - Non-instance trading
    - Multi-period price impact
- Active and Excess Return
- Single-Period Optimisation
    - SPO Notation
    - Optimization
    - Risk Function $\psi_t$
        - Transformed risk
        - Worst-case quadratic risk
    - Forecast Error Risk
        - Return Forecast Error
- Other Constraints
    - $\beta$ Neutral
    - Factor Neutral
    - Stress Constraints
    - Non-convexity
- Multi-Period Optimization
    - Terminal Constraints

<!-- /MarkdownTOC -->


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

$F_t \in \mathbb{R}^{(n+1) \times k}$ is the factor exposure matrix at time $t$.

$F^T_{r_t}$ - factor return matrix.

$\Sigma^f_t \in \mathbb{R}^{k \times k}$ is the factor covariance matrix of $F^T_{r_t}$.

Factor covariance matrix: $\Sigma_t = F_t \Sigma^f_t F^T_t + D_t$, where $D_t \in \mathbb{R}^{(n+x)\times(n+1)}$ is a non-negative diagonal matrix aka. **idiosyncratic risk**.

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

### SPO Notation

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

This essentially says the trades plus t-cost and funding costs should not impact cash position. This constraint in this format is **non-convex**. We can **simplify** this constraint to just:

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

#### Return Forecast Error

Can be modelled as Worst-case quadratic risk. I.e. given return forecast confidence interval $\rho$, we can have a risk model of:

$$ \psi_t (x) = \rho^T |x - w^b_t| $$

## Other Constraints

### $\beta$ Neutral

$$ (w^b_t)^T \Sigma_t (w_t + z_t) = 0 $$

### Factor Neutral

For a factor model, the estimated portfolio risk $\sigma^F_i$ for factor $i$ is given by:

$$ \big( \sigma^F_i \big)^2 = (w_t + z_t)^T (F_t)_i (\Sigma^{f}_{t})_{ii} (F_t)^T_i (w_t + z_t) $$

The constraint for the portfolio to be neutral for factor $i$ means $\sigma^F_i = 0$, hence:

$$ (F_t)^T_i (w_t + z_t) = 0$$

### Stress Constraints

I.e. setting minimum **expected** return, $R^{min}$, for a set of scenarios.

For each scenario $i$ with **predicted** return $c_i$:

$$ c^T_i (w_t + z_t) \geq R^{min} $$

### Non-convexity

Some tricks for working around non-convex constraints.

Avoid **small** trades, i.e. $|(z_t)_i| \geq \epsilon \; \text{for} \; (z_t)_i \neq 0$, can either use cost terms or the following:

Solve SPO without this constraint, finding solution $\tilde{z}$. Based on this result, we then set a set of zero, negative, or positive constraints. Basically, for small trades in $\tilde{z}$, we set $(z_t)_i = 0$. This works well in practice.

Another example is to solve for an max $K$ no. of trades problem. First, run SPO without the constraint. Find the largest $K$ trades, and set constraints to trade only those assets.

## Multi-Period Optimization

Define a horizon of $H$ time periods as:

$$ t, t+1, \cdots, t + H - 1 $$

Solve for trades for these periods: $z_t, z_{t+1}, \cdots, z_{t+H-1}$.

$\hat{r}_{\tau | t}$ denotes return forecast predicted for time period $\tau$ at time period $t$. E.g. $\hat{r}_{t+2 | t}$ is the return forcast for time period $t+2$, predicted at time $t$.

Natural objective is to maximize the sum of expected risk-adjusted return for this horizon:

$$ \max \sum^{t+H-1}_{\tau = t} \big( \hat{r}^T_{\tau | t} w_\tau + \hat{r}^T_{\tau | t} z_\tau - \hat{\phi}^{trade}_\tau(z_\tau) - \hat{\phi}^{hold}_\tau (w_\tau + z_\tau) - \gamma_\tau \psi_\tau (w_\tau + z_\tau) \big) $$

Here, $w_t$ is known (current portfolio weights), but $w_{t+1}, \cdots, w_{t+H-1}$ are not known, without these we **cannot** specify the objective. We need a simplification / approximation below.

The paper suggests that we assume $r_t = 0$, and therefore $w_{t+1} = w_t + z_t$. This is a reasonable assumption if the period returns are small, i.e. we are ignoring *second-order* terms. This is a key assumption that allows us to propagate the portfolio forward.

We must add constraint $1^T z_t = 0$ to ensured that weights sum to 1, i.e.

$$ 1^T w_\tau = 1, \; \tau = t + 1, \cdots, t+H $$

Otherwise, if $1^T w_t = 1$ and $1^T z_t \neq 0$, then in this approximation, we have $1^T w_{t+1} \gt 1$.

Therefore, we need:

$$ 1^T z_\tau = 0, \; \tau = t + 1, \cdots, t+H-1 $$

With the assumption $w_{\tau+1} = w_\tau + z_\tau$, the MPO problem can be written as:

$$
\begin{aligned}
\max \;\; &\sum^{t+H-1}_{\tau = t} \big( \hat{r}^T_{\tau | t} (w_\tau + z_\tau) - \hat{\phi}^{trade}_\tau(z_\tau) - \hat{\phi}^{hold}_\tau (w_\tau + z_\tau) - \gamma_\tau \psi_\tau (w_\tau + z_\tau) \big) \\
\text{subject to} \;\; &1^T z_\tau = 0 \\
&z_\tau \in \mathcal{Z_\tau} \\
&w_\tau + z_\tau \in \mathcal{W_\tau} \\
&w_{\tau+1} = w_\tau + z_\tau \\
&\tau = t, t+1, \cdots, t + H - 1
\end{aligned}
$$

Same notation as SPO here:

- $\mathcal{Z_\tau}$ - trade constraints at period $\tau$
- $\mathcal{W_\tau}$ - holding constriants at period $\tau$

We ignore $\hat{r}^T_{t | t} w_t$ since $\hat{r}^T_{t | t} = constant$, return from time $t$ to $t$.

We can futher eliminate $z_\tau$ given $w_{\tau + 1} = w_\tau + z_\tau$, in the objective, i.e.:

$$ \max \;\; \sum^{t+H-1}_{\tau = t} \big( \hat{r}^T_{\tau | t} w_{\tau + 1} - \hat{\phi}^{trade}_\tau(z_\tau) - \hat{\phi}^{hold}_\tau (w_{\tau + 1}) - \gamma_\tau \psi_\tau (w_{\tau + 1}) \big) $$

Then by shifting start from $t$ to $t+1$:

$$
\begin{aligned}
\max \;\; &\sum^{t+H}_{\tau = t+1} \big( \hat{r}^T_{\tau | t} w_\tau - \hat{\phi}^{trade}_\tau(w_\tau - w_{\tau-1}) - \hat{\phi}^{hold}_\tau (w_\tau) - \gamma_\tau \psi_\tau (w_\tau) \big) \\
\text{subject to} \;\; &1^T w_\tau = 1 \\
&w_\tau - w_{\tau-1} \in \mathcal{Z_\tau} \\
&w_\tau \in \mathcal{W_\tau} \\
&\tau = t+1, \cdots, t + H
\end{aligned}
$$

Here, note that if we shift to $t+1$ start, we still have to solve for trades $z_t$. Therefore, $\hat{\phi}^{trade}_\tau(z_\tau)$ becomes $\hat{\phi}^{trade}_\tau(z_{\tau-1})$, and $z_{\tau-1} = w_\tau - w_{\tau-1}$. Also, we do not need $z_{t+H}$.

### Terminal Constraints

When we have a reasonably long horizon, we can add a terminal equality constraint. A reasonable example would be to converge to benchmark weights:

$$ 1^T w_{t+H} = w^b $$
