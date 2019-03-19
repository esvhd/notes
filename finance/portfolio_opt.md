# Portfolio Optimization

Some notes based on Active Portfolio Management (APM) 2nd edition by Grinold and Kahn.

For basic vol metrics for portfolio, check out my code for risk parity. 


## CAPM

$r_p(t)$ - Portfolio $p$ excess returns for time period $t$.

$r_M(t)$ - market portfolio excess returns.

Excess return defined as return over risk free asset for the same time period.

**Assume** that we know the beta $\beta_p$ of a portfolio $p$ versus the market portfolio $M$, we can decompose portfolio return into two parts:

$$ r_p = \beta_p \times r_M + \theta_p $$

$\theta_p$ is the **residual return**, **assumption** is that this residual return is **uncorrelated** with market return $r_M$. This is clearly a **strong** assumption in practice. With this assumption, the **variance for portfolio** $p$ is

$$\sigma^2_p = \beta^2_p \sigma^2_M + w^2_p $$

$w^2_p$ is the **residual variance** of portfolio $p$, i.e. the variance of $\theta_p$.

CAPM states that the expected residual retrn on all assets and any portfolio is equal zero, i.e. $E[\theta_p] = 0$.

If we start with a market portfolio and assume it's optimal, we can back the the exact CAPM expected returns.

If we plot CAPM derived stock or portfolio expected returns (y-axis) against their beta vs market portfolio (x-axis), they will lie on the **security market line**, with the intercept at risk free rate.

For ex-post / realized returns, can be plotted in the same way, the deviation from the security market line represent the skills of the manager.

CAPM expected return forecasts will only be as good as the estimate of beta.

## Risk Models

$h_p$ - portfolio asset weights, $N \times 1$

$h_B$ - benchmark asset weights, $N \times 1$

$h_{PA} = h_p - h_B$ - active weights

$N$ - no. of assets

$X$ - asset factor exposure matrix, $N \times K$ matrix

$b$ - factor returns, shape $K \times 1$

$u$ - specific returns, shape $N \times 1$

Hence we decompose return as: $r = X \times b + u$. (shape: $N \times K \cdot K \times 1 = N \times 1$)

$F$ - factor covariance matrix, $K \times K$

$\Delta$ - specific return variance, $N \times N$ diagonal matrix.

Hence $N \times N$ covariance matrix $V$ is $V = X \cdot F \cdot X^T + \Delta$

Portfolio factor exposure: $x_p = X^T \cdot h_p$, shape is $K \times N \cdot N \times 1 = K \times 1$

Portfolio factor variance: 

$$
\begin{aligned}
\sigma^2_p &= x^T_p \times F \times x_p + h^T_p \times \Delta \times h_p \\
&= h^T_p \cdot V \cdot h_p
\end{aligned}
$$

Portfolio **active** factor exposure: $x_{PA} = X^T \cdot h_{PA}$, shape $K \times N \cdot N \times 1 = K \times 1$

Active Risk / Tracking Error:

$$
\begin{aligned}
\Psi^2_p &= x^T_{PA} \cdot F \cdot x_{PA} + h^T_{PA} \cdot \Delta h_{PA} \\
&= h^T_{PA} \cdot V \cdot h_{PA}
\end{aligned}
$$

In both portfoli variance and trackig error, we broken them down into factor and specific risks. This works only with the assumption that factor risk and specific risk are **uncorrelated**.

Asset level beta, $N \times 1$ vector, is:

$$\beta = \frac{V \cdot h_B}{\sigma^2_B}$$

Portfolio beta, shape $1 \times 1$:

$$\beta_p = h^T_p \cdot \beta$$

The APM book p77 has alternative definitions for beta which breaks them down into factor and specific components.

Active beta (my understnding): $\beta_{PA} = h^T_{PA} \cdot \beta$.

Residual Covariance matrix $VR$ is given by: $VR = V - \beta \cdot \sigma^2_B \cdot \beta^T$. Shape is $N \times N - N \times 1 \cdot 1 \times 1 \cdot 1 \times N = N \times N$
