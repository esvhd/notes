# Portfolio optimisation

Some notes based on Active Portfolio Management (APM) 2nd edition by Grinold and Kahn.

For basic vol metrics for portfolio, check out my code for risk parity. 


# CAPM

$r_p(t)$ - Portfolio $p$ excess returns for time period $t$.

$r_M(t)$ - market portfolio excess returns.

Excess return defined as return over risk free asset for the same time period.

**Assume** that we know the **beta** $\beta_p$ of a portfolio $p$ versus the market portfolio $M$, we can decompose portfolio return into two parts:

$$ r_p = \beta_p \times r_M + \theta_p $$

$\theta_p$ is the **residual return**, **assumption** is that this residual return is **uncorrelated** with market return $r_M$. This is clearly a **strong** assumption in practice. With this assumption, the **variance for portfolio** $p$ is

$$\sigma^2_p = \beta^2_p \sigma^2_M + w^2_p $$

$w^2_p$ is the **residual variance** of portfolio $p$, i.e. the variance of $\theta_p$, aka. $Var(\theta_p)$. Its shape is $1\times1$

CAPM states that the expected residual retrn on all assets and any portfolio is equal zero, i.e. $E[\theta_p] = 0$.

If we start with a market portfolio and assume it's optimal, we can back the the exact CAPM expected returns.

If we plot CAPM derived stock or portfolio expected returns (y-axis) against their beta vs market portfolio (x-axis), they will lie on the **security market line**, with the intercept at risk free rate.

For ex-post / realized returns, can be plotted in the same way, the deviation from the security market line represent the skills of the manager.

CAPM expected return forecasts will only be as good as the estimate of beta.

# Risk Models

$h_p$ - portfolio asset weights, $N \times 1$

$h_B$ - benchmark asset weights, $N \times 1$

$h_{PA} = h_p - h_B$ - active weights

$r_p$ - portfolio return

$r_B$ - benchmark return

$r_{PA}$ - portfolio active return, $r_p - r_B$

$\beta_p$ - portfolio beta to benchmark

$N$ - no. of assets

$X$ - asset factor exposure matrix, $N \times K$ matrix

$b$ - factor returns, shape $K \times 1$

$u$ - specific returns, shape $N \times 1$

Hence we decompose return as: $r = X \times b + u$. (shape: $N \times K \cdot K \times 1 = N \times 1$)

$F$ - factor covariance matrix, $K \times K$

$\Delta$ - specific return variance, $N \times N$ diagonal matrix.

$V$ - $N \times N$ covariance matrix, $V = X \cdot F \cdot X^T + \Delta$

Portfolio factor exposure: $x_p = X^T \cdot h_p$, shape is $K \times N \cdot N \times 1 = K \times 1$

Portfolio factor variance, $1 \times 1$ shape: 

$$
\begin{aligned}
\sigma^2_p &= x^T_p \times F \times x_p + h^T_p \times \Delta \times h_p \\
&= h^T_p \cdot V \cdot h_p
\end{aligned}
$$

Portfolio **active** factor exposure: $x_{PA} = X^T \cdot h_{PA}$, shape $K \times N \cdot N \times 1 = K \times 1$

**Active Risk / Tracking Error**, $\Psi_p$:

$$
\begin{aligned}
\Psi^2_p &= x^T_{PA} \cdot F \cdot x_{PA} + h^T_{PA} \cdot \Delta \cdot h_{PA} \\
&= h^T_{PA} \cdot V \cdot h_{PA}
\end{aligned}
$$

$\Psi_p$ has shape $1 \times K \cdot K \times K \cdot K \times 1 + 1 \times N \cdot N \times N \cdot N \times 1 = 1 \times 1$

In both portfolio variance and trackig error, we broken them down into factor and specific risks. This works only with the assumption that factor risk and specific risk are **uncorrelated**.

**Active Returns vs Residual Returns**

Active return is the difference between portfolio return and benchmark return:

$$ r_{PA} = r_p - r_B = \theta_p + \beta_{PA} \times r_B $$

Active risk is defined as:

$$ \Psi_p = \sqrt{r_{PA}} = \sqrt{w^2_p + \beta^2_{PA} \times \sigma^2_B} $$

**Active return = residual return** when the manager avoids benchmark timing and set $\beta_p = 1$.

**Asset level beta**, $N \times 1$ vector, is:

$$\beta = \frac{V \cdot h_B}{\sigma^2_B}$$

**Portfolio beta**, shape $1 \times 1$:

$$\beta_p = h^T_p \cdot \beta$$

The APM book p77 has alternative definitions for beta which breaks them down into factor and specific components.

**Active beta** (my understnding): $\beta_{PA} = h^T_{PA} \cdot \beta$.

**Residual Covariance** matrix $VR$ is given by: $VR = V - \beta \cdot \sigma^2_B \cdot \beta^T$. Shape is $N \times N - N \times 1 \cdot 1 \times 1 \cdot 1 \times N = N \times N$

## Risk Attribution

### Position Marginal Contribution

This section computes each position's risk contribution.

Marginal Contribution to Total Risk, **MCTR**, change in $\sigma_p$ for a $1\%$ change in holdings $h_p$:

$$ MCTR = \frac{\partial\sigma_p}{\partial h^T_p} = \frac{V \times h_p}{\sigma_p} $$

Shape is $(N \times N \cdot N \times 1 ) / (1 \times 1) = N \times 1$

$h_{PR} = h_p - \beta_p \times h_B$ - **residual holding vector**, shape $N \times 1$

Marginal Contribution to Residual Risk, **MCRR**, with hape $N \times 1$:

$$ MCRR = \frac{VR \times h_p}{w_p} = \frac{V \times h_{PR}}{w_p} $$

Marginal Contribution to Active Risk, **MCAR**, with shape $N \times 1$:

$$ MCAR = \frac{V \times h_{PA}}{\Psi_p} $$


### Factor Marginal Contribution

This sectionc computes each factor's risk contribution. See book for details.


# Exceptional Return / Value Added

**Exceptional Return** is the difference between forecast / predicted return and concensus return.

Active management **value-added** is the expected Exceptional Returns mintues a penalty of active variance.

Expected return can be decomposed into **four parts**:

$$ E[R_n] = 1 + i_F + \beta_n \times \mu_B + \beta_n \times \Delta f_B + \alpha_n $$

These terms are:

1. Time Preimum, $i_F$
2. Risk Premium, $\beta_n \times \mu_B$
3. Exceptional Benchmark Returns, $\beta_n \times \Delta f_B$, difference between expected excess return on benchmark and long term expected excess return.
4. $\alpha$ is the expected residual return

$\mu_B$: expected excess return on benchmark, typically a long run (70+ year) average.

$r_B$ - benchmark excess returns over risk free asset.

$r_n$ - excess return for asset $n$

$\beta_n$ - asset $n$'s **excess return beta** vs the benchmark excess return. Defined as:

$$ \beta_n = \frac{Cov(r_B, r_n)}{Var(r_B)} $$

$R_n$ - total return for asset $n$

$f_B$ - expected benchmark excess returns, $f$ means forecast

$\Delta f_B = f_B - \mu_B$

Forecast expected return for asset $n$ is $f_n = \beta_n \times f_B + \alpha_n$.

## Utility Function

A simple utility function to maximize is to trade residual return versus residual risk. E.g. 

$$ U_p = \alpha_p - \lambda_R \times w^2_p $$

Where $\lambda_R$ is a measure of risk aversion to residual risk.

# Information Ratio

Defined as: 

$$ IR_p = \frac{\alpha_p}{w_p} $$

$IR$ measures the **annual** $\alpha_p$ (residual return) relative to $w_p$ (stdev of residual return).

By definition, benchmark and risk free assets have 0 IR, since they have 0 $\alpha_p$.

Subsitituting into the utility function above:

$$ U_p = w_p \times IR_p - \lambda_R \times w^2_p $$

To maximize utility, set $\partial{U_p}/\partial{w_p} = 0$, the **optimal residual risk** $w_p$, denoted by $w^*$ is:

$$ w^* = \frac{IR_p}{2\lambda_R} $$

The following simple formula gives an **approximation** of the $IR$:

$$ IR \approx IC \times \sqrt{BR} $$

$IC$ - **Information Coefficient**, measure the **correlation** between predicted returns and realized returns.

$BR$ - **Breath**, defined as the number of independent predictions of **exceptional returns** made per year. See p328, $BR$ is more difficult to measure than either $IC$ or $IR$.

See derivation of this and assumptions made in Grinold & Kahn p166-p168.

Hence, with the above **optimal residual risk** formula:

$$ w^* = \frac{IR}{2\lambda_R} = \frac{IC \times \sqrt{BR}}{2\lambda_R} $$

Therefore, the desired level of risk has a linear relationship with $IR$ and $\sqrt{BR}$.

The Value Added is:

$$ VA^* = \frac{IR^2}{4\lambda_R} = \frac{IC^2 \times BR}{4\lambda_R} $$

$VA$ has a linear relationship with $IC^2$ and $BR$.

If instead of predicting expected exceptional returns, we predict the sign of exceptional returns, we have:

$$
\begin{aligned}
IC &= \frac{1}{N}\big[N_1 - (N - N_1)\big] = 2 \bigg(\frac{N_1}{N}\bigg) - 1 \\
&= Accuracy - (1 - Accuracy) \\
&= 2 \times Accuracy - 1
\end{aligned}
$$

Where:

* $N$ is the total number of predictions, $P + N$, no. of positive and negative classes
* $N_1$ is the number of correct predictions, $TP + TN$, true positive and true negatives.

This is reminds me of the binary classification loss for class labels $[0, 1]$.

$$ \mathcal{L} = \frac{1}{N}\sum^N_{i=1}\big[\hat{y}_i (1 - y_i) + (1 - \hat{y}_i) y_i \big] $$

Where:

* $y_i$ is the ground truth for sample $i$
* $\hat{y}_i$ is the predicted label for sample $i$

Therefore, we can compare the two metric, correlation and hit ratio between forecasts. 

## Additivity

$$IR^2 = IC^2_1 \times BR_1 + IC^2_2 \times BR_2 $$

Big **assumption** here is that each IC is derived from independent information, i.e. no correlated predictions between $IC_1$ and $IC_2$.

## Assumptions

1. Sources of information should be **independent**. Don't bet on the same information twice. 

Dependency over time is also an issue. E.g. quarterly predictions should be based on **new** information from each quarter.

For the additivity example above, if 2 strategies each have skill IC, but information correlation is $\gamma$, then the combined IR is:

$$ IR_{combined} = IC^2 \times \sqrt{\frac{2}{1 + \gamma}} $$

2. The law assumes that each of the $BR$ active bets has the **same level of skill**. If the levels of skill measured by $IC$ are different, and we plot $BR$ vs $IC^2$, then the overall $IC$ is the area under this curve.

3. The strongest assumption behind the law is that the manager will build portfolios that use the information in the optimal way, utilising its value.

Other practical portfolio constraints such as short sale would result in a drop in realized $IR$.

## Performance Evaluation

When comparing portfolios, we can estimate the $\alpha$ of a portfolio with OLS regression. $\alpha$ in this case is the **intercept** of the OLS formula. 

$$ r(t) = \alpha + \beta \times r_B(t) + \epsilon(t) $$

In the context of traditional statistical testing, $t$-statistics is then:

$$ \text{t-stat} = \frac{\alpha}{stdev(\alpha)} $$

This links it back to $IR$ which uses **annualised** $\alpha$. 

$$ IR \approx = \frac{\text{t-stat}}{\sqrt{T}} $$

Where $T$ is a floating scaler of no. of years.

# Portfolio Construction

Portfolio constrained optimisation objective, for $N$ assets:

$$ \underset{h_{p, i} > 0, |h_{PA, i} |< 5\% \forall i \in N}{\operatorname{argmax}} h^T_{PA} \alpha' - \lambda' \cdot h^T_{PA} \cdot V \cdot h_{PA} $$

Constraints are:

* no short sale
* active position deviation from benchmark < 5%. This number can change.
* See $\Psi^2_{p}$ definition in sections above.

The **modified alpha** is:

$$ \alpha' = \bigg(\frac{IR}{\Psi^*_p}\bigg) \times V \times h^*_{PA} $$

Appropriate active risk aversion is:

$$\lambda' = \frac{IR}{2 \times \Psi^*_p} $$

Alpha has a **scale**, $stdev(\alpha) \sim \sigma \times IC$. If they don't have the right scale, scale alpha before use.

Outsized alpha estimates can have undue influence, they shoud be **trimmed**.

## Dispersion

The concept of dispersion measures the performance difference between the best and worst performing separate account mandates from a manager. 

If alpha and risk stay absolutely **constant** over time, then dispersion persists. Then the remaining tracking error is bounded by:

1. transaction cost, $TC$
2. manager's risk aversion, $\lambda_A$

Specifically:

$$ \Psi^2 \leq \frac{TC}{2 \times \lambda_A} $$

In practice, alpha and risk **vary** over time, then convergence of performance will occur. However, this is a general argument and **does not** imply any particular time scale.

Dual-benchmark optimisation - essentially trying to balance two objectives - would trade off return versus dispersion. 

## Alpha Analysis

My thoughts on this is that the material and maths used hevily rely on the assumption that factor returns and specific returns are **uncorrelated**, and that components of specific returns are also **uncorrelated**. 

These assumption most likely won't hold in practice especially looking at a corporate bond or equity market risk model. 

The technical appendix also has a section on the impact of **covariance matrix estimation error**. Worth a read.
