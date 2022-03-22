# Portfolio Optimisation

Some notes based on Active Portfolio Management (APM) 2nd edition by Grinold and Kahn.

For basic vol metrics for portfolio, check out my code for risk parity.

- [Portfolio Optimisation](#portfolio-optimisation)
- [CAPM](#capm)
- [Risk Models](#risk-models)
  - [Risk Attribution](#risk-attribution)
    - [Position Marginal Contribution](#position-marginal-contribution)
    - [Factor Marginal Contribution](#factor-marginal-contribution)
- [Exceptional Return / Value Added](#exceptional-return--value-added)
  - [Utility Function](#utility-function)
    - [Utility and $\alpha$ 2nd Edition {#utility}](#utility-and-alpha-2nd-edition-utility)
  - [Value Added](#value-added)
- [Information Ratio](#information-ratio)
  - [Additivity](#additivity)
  - [Assumptions](#assumptions)
  - [Performance Evaluation](#performance-evaluation)
- [Portfolio Construction](#portfolio-construction)
  - [Dispersion](#dispersion)
  - [Alpha Analysis](#alpha-analysis)
- [Performance Attribution Models](#performance-attribution-models)
  - [Brinson Hood Beebower (BHB) Attribution](#brinson-hood-beebower-bhb-attribution)
  - [Brinson-Fachler (BF) Model](#brinson-fachler-bf-model)
- [Maximum Sharpe Ratio Portfolio](#maximum-sharpe-ratio-portfolio)
- [2nd Edition](#2nd-edition)
  - [Breath](#breath)
  - [Transfer Coefficient](#transfer-coefficient)
  - [Dynamic Portfolio Management](#dynamic-portfolio-management)
    - [Notation](#notation)
    - [Signal Weighting](#signal-weighting)

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

$X$ - asset factor exposure matrix, $N \times K$ matrix. Typically generated
with a time series regression of `asset_return_t ~ factor_return_t` for `t`
time period. With this method, the factor exposure generated measures the
**average** exposure over the regression period.

$b$ - factor returns, shape $K \times 1$

$u$ - specific returns, shape $N \times 1$

Hence we decompose return as: $r = X \times b + u$. (shape: $N \times K \cdot K \times 1 = N \times 1$)

$F$ - factor covariance matrix, $K \times K$

$\Delta$ - specific return variance, $N \times N$ diagonal matrix, assumption here is they are uncorrelated.

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

In both portfolio variance and trackig error, we break them down into factor and specific risks. This works only with the **assumption** that factor risk and specific risk are **uncorrelated**.

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

Marginal Contribution to Residual Risk, **MCRR**, with shape $N \times 1$:

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

$$ U_p = \alpha_p - \lambda_R \times w^2_p = f_p - \lambda_T \times \sigma^2_p$$

Where $\lambda_R$ is a measure of **risk aversion** to residual risk.

When we have **no information**, $f_p = \mu_B$, with rearranging the utility function above, we see that the level of risk aversion that leads us to choose the benchmark is:

$$ \lambda_T = \frac{\mu_B}{2\sigma^2_B} $$

E.g. for $\mu_B = 6\%$ and $\sigma_B = 20\%$, we have $\lambda_T = .06 / (2 \times .2^2) \approx .75$.

### Utility and $\alpha$ 2nd Edition {#utility}

In the recent edition of the book, we connect utility and $\alpha$ as:

$$
\begin{aligned}
U &= h^T \cdot \alpha - \lambda h^T \cdot V \cdot h \\
&= \alpha_p - \lambda w^2_p
\end{aligned}
$$

Where $\lambda$ is a risk-aversion parameter capturing investor preferences.

To find the maximum of $U$, we take the derivate w.r.t. $h$:

$$
\begin{aligned}
\frac{\partial U}{\partial h} &= \alpha - 2 \lambda \cdot V \cdot h = 0\\
\alpha_p &= 2 \lambda \cdot V \cdot h \\
h_{max} &= \frac{\alpha}{2\lambda \cdot V}
\end{aligned}
$$

From the optimal portfolio would vary with risk aversion parameter.

## Value Added

Risk and return can be split into three parts:

* **Intrinsic**, $f_B - \lambda_T \times \sigma^2_B$, comes from benchmark return.
* **Timing**, $VA_{t} = \beta_{PA}\times \Delta f_B - \lambda_{BT} \times \beta^2_{PA} \times \sigma^2_B$, a manager's **active beta**
* **Residual**, $VA_{r} = \alpha_p - \lambda_R \times w^2_p$, due to manager's residual positions.

In other words, **value added** is the sum of last two parts above, i.e., returns from active beta and residual positions.

The objective of active management is to **maximise value added**, defined as:

$$ VA = \big( \beta_{PA} \times \Delta f_B - \lambda_{BT} \times \beta^2_{PA} \times \sigma^2_B \big) + \big(\alpha_p - \lambda_R \times w^2_p \big)$$

To **maximize benchmark timing**:

$$ \frac{\partial{VA_t}}{\partial \beta_PA} = \Delta f_B - 2 \lambda_{BT} \cdot \beta_{PA} \cdot \sigma^2_B $$

Hence the **optimal** level of $\beta_{PA}$ is:

$$\beta^*_{PA} = \frac{\Delta f_B}{2 \cdot \lambda_{BT} \cdot \sigma^2_B} $$

Active beta equals zero when:

* No benchmark forecast, i.e. $\Delta f_B = 0$
* High level or benchmark timing risk aversion, i.e. high $\lambda_{BT}$

Looking at the value added for benchmark timing, $VA_t$:

$$
\begin{aligned}
VA_t &= \beta^*_{PA} \times \Delta f_B - \lambda_{BT} \cdot (\beta^*_{PA})^2 \cdot \sigma^2_B \\
&= \frac{(\Delta f_B)^2}{2 \cdot \lambda_{BT} \cdot \sigma^2_B} - \lambda_{BT} \cdot \sigma^2_{B} \frac{(\Delta f_B)^2}{4 \cdot \lambda^2_{BT} \cdot \sigma^4_B} \\
&= \frac{2 (\Delta f_B)^2 - (\Delta f_B)^2}{4 \cdot \lambda_{BT} \cdot \sigma^2_B} \\
&= \frac{(\Delta f_B)^2}{4 \cdot \lambda_{BT} \cdot \sigma^2_B}
\end{aligned}
$$

We can use this formula to assess the impact of benchmark timing, e.g. assuming $\sigma_B = 17\%$, $\Delta f_B = 4\%$, with median risk aversion $\lambda_{BT} = 9.$, $VA_t = 0.04^2 / (4 \times 9 \times 0.17^2) \approx 0.00154$. The book is not clear here about the unit of $VA$. In the text it refers to this as $15.4$bp, both $VA$ and $\beta^*_{PA}$ are scaled by $1e-2$...

Finance text is really not precise when it comes to maths and units. In the text, in calculations the authors used 4 to represent 4% return... This means that $\lambda_{BT}$ here is in different scale. To follow the more accurate way of representing 4% as 0.04, $lambda_{BT}$ in the book needs to be scaled up by 100.

# Information Ratio

Defined as:

$$ IR_p = \frac{\alpha_p}{w_p} $$

$IR$ measures the **annual** $\alpha_p$ (residual return) relative to $w_p$ (stdev of residual return).

By definition, benchmark and risk free assets have 0 IR, since they have 0 $\alpha_p$.

Subsitituting into the utility function above (intuitively, maximise return / alpha vs risk):

$$ U_p = w_p \times IR_p - \lambda_R \times w^2_p $$

To maximize utility, set $\partial{U_p}/\partial{w_p} = 0$, the **optimal residual risk** $w_p$, denoted by $w^*$ is:

$$ w^* = \frac{IR_p}{2\lambda_R} $$

The following simple formula gives an **approximation** of the $IR$:

$$ IR \approx IC \times \sqrt{BR} $$

$IC$ - **Information Coefficient**, measure the **correlation** between predicted returns and realized returns. In finance an $IC$ of 0.05 is good, 0.10 would be seen as great.

$BR$ - **Breath**, defined as the number of independent predictions of **exceptional returns** made per year. See p328, $BR$ is more difficult to measure than either $IC$ or $IR$. Note also that it is a **rate**, not a number, so the no. of assets in a portfolio isn't the right measure.

See derivation of this and assumptions made in Grinold & Kahn p166-p168.

Hence, with the above **optimal residual risk** formula:

$$ w^* = \frac{IR}{2\lambda_R} = \frac{IC \times \sqrt{BR}}{2\lambda_R} $$

Therefore, the desired level of risk has a **linear relationship** with $IR$ and $\sqrt{BR}$.

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

# Performance Attribution Models

## Brinson Hood Beebower (BHB) Attribution

Given portfolio of assets $i \in 1, \cdots, N$, $P$ to denote portfolio, $B$ to denote benchmark:

- individual asset weights - $w^P_i$, $w^B_i$ for portfolio and benchmark
- asset returns $r_i$
- portfolio sector weight $W^P_j = \sum_{i \in j} w^P_i$
- sector average return $R^P_j = \frac{\sum_{i \in j} w^P_i r_i}{W^P_j}$
- $R^{B}_j$ - benchmark return for sector $j$
- $R^{P}_j$ - portfolio return for sector $j$
- portfolio return $R^P = \sum_{i} w^P_j r_i = \sum_{j} W^P_j R^P_j$
- benchmark return $R^B = \sum_{i} w^B_j r_i = \sum_{j} W^B_j R^B_j$

Therefore, we have Brinson attribtion as follows:

- Active return $R_{active} = R^P - R^B = \sum_{j} W^P_j R^P_j - \sum_{j} W^B_j R^B_j$
- Allocation effect $R_{allocation} = \sum_j (W^P_j - W^B_j) R^B_j$ - bench return is the baseline, allocation shows up in sector weights
- Selection effect $R_{selection} = \sum_j W^B_j (R^P_j - R^B_J)$ - selection is reflected in sector average return, weighted by bench weight
- Interaction effect $R_{interaction} = R_{active} - R_{allocation} - R_{selection} = \sum_j (W^P_j - W^B_j)(R^P_j - R^B_j)$

**Drawback of BHB**

1. the number of terms grows expoentially with the no. of groups. Hard to understand beyond 2 groups.
2. For a sector with positive return, overweight in portfolio would generate positive allocation effect,
  **even if the sector return is lower than overall benchmark**

## Brinson-Fachler (BF) Model

BF model is designed to address BHB's drawback no. 2.

**Drawback for BF** is still that with different ways of grouping portfolios, allocation / selection effect can be
very different.

- Allocation effect $R_{allocation} = \sum_j (W^P_j - W^B_j) (R^B_j - R^B)$
- No change to other terms

To derive this, **provided there is no leverage**, we have:
$$
\begin{aligned}
\sum_j W^P_j = \sum_j W^B_j &= 1 \\
\therefore \sum_j W^P_j - 1 &= 0 \\
\sum_j W^P_j - \sum_j W^B_j &= 0 \\
\therefore \sum_j (W^P_j - W^B_J) &= 0
\end{aligned}
$$

Start with BHB $R_{allocation}$, given $R^B$ is a constant, we have:
$$
\begin{aligned}
R_{allocation} &= \sum_j (W^P_j - W^B_j) R^B_j - \sum_j (W^P_j - W^B_j) R^B + \sum_j (W^P_j - W^B_j) R^B \\
&= \sum_j (W^P_j - W^B_j)(R^B_j - R^B) + \sum_j (W^P_j - W^B_j) R^B \\
&= \sum_j (W^P_j - W^B_j)(R^B_j - R^B) + R^B \sum_j (W^P_j - W^B_j)\\
&= \sum_j (W^P_j - W^B_j)(R^B_j - R^B) + R^B \times 0\\
&= \sum_j (W^P_j - W^B_j)(R^B_j - R^B)
\end{aligned}
$$

If leverage is allowed, then the allocation effect should have another term, i.e. $+\sum_j (W^P_j - W^B_j) R^B$.

# Maximum Sharpe Ratio Portfolio

$N$ assets with return vector R, and covariance matrix $\sigma^2$, weight vector $w$

portfoio return: $R_p = w^T R$

portfolio risk: $\sigma^2_p = w^T \sigma^2 w$

So sharpe ratio is $S_p = \frac{R_p}{\sigma_p}$

To find the max we solve for $\frac{\partial{S_p}}{\partial w} = 0$.

Assume 2 assets, and covariance is $\sigma_{1,2}$, we have:

$$ S_p = \frac{w R_1 + (1-w) R_2}{\sqrt{w^2 \sigma^2_1 + (1-w)^2 \sigma^2_2 + 2w(1-w)\sigma_{1,2}}} $$

Thanks for the calcs in this [video](https://www.youtube.com/watch?v=IhYhVW6IO7I),
the optimial weight is:

$$ w^* = \frac{R_1 \sigma^2_2 - R_2 \sigma_{1,2}}{R_1 \sigma^2_2 + R_2 \sigma^2_1 - (R_1 + R_2)\sigma_{1,2}} $$

# 2nd Edition

## Breath

Given an **information turnover rate**, $\gamma$, that captures both old info decay rate and new info arrival rate. When these two processes are in balance,
we show that the breath of this forecast for $N$ assets is:

$$ BR = \gamma \cdot N $$

## Transfer Coefficient

p52 of the book.

The transfer coefficient measures the correlation between the return of an optimal portfolio without constraints and t-costs, and the actual portfolio that is run.
It is a measure of **implementation efficiency**.

With reference to the [Utility](#utility-and-alpha-2nd-edition-utility) section,
given an optimal portfolio, $Q$, we have:

$$ \alpha - 2 \lambda \cdot V \cdot h_Q = 0 $$

The total alpha for this portfolio is: $\alpha_Q = h_Q^T \cdot \alpha$.
Subsitituting $\alpha$ from above, we have:

$$
\begin{aligned}
\alpha_Q &= 2 \lambda \cdot h_Q^T \cdot V \cdot h_Q \\
&= 2 \lambda \sigma_Q^2
\end{aligned}
$$

Where $\sigma_Q^2$ is the portfolio residual variance for $Q$. In other parts of this note, we also use $\omega$ to represent residual variance.

The information ratio therefore is:

$$ IR_Q = \frac{\alpha_Q}{\sigma_Q} = 2 \lambda \cdot \sigma_Q $$

In practice, investors don't hold this optimal portfolio due to constraints,
they hold portfolio $P$ instead. Therefore, by subsitituting $\alpha$ from
earlier we have:

$$
\begin{aligned}
\alpha_P &= h_P^T \cdot \alpha \\
&= 2 \lambda \cdot h_P^T \cdot V \cdot h_Q \\
&= 2 \lambda \cdot Covariance(P, Q) \\
&= 2 \lambda \cdot \sigma_P \cdot \sigma_Q \cdot \rho_{PQ}
\end{aligned}
$$

Where $\rho_{PQ}$ is the correlation of portfolio $P$ and $Q$. Also, from
standard Pearson correlation coefficient formula:

$$ \rho_{PQ} = \frac{Cov(P, Q)}{\sigma_P \sigma_Q} \therefore Cov(P, Q) = \sigma_P \cdot \sigma_Q \cdot \rho_{PQ} $$

Therefore, the information ratio for $P$ is:

$$ IR_P = \frac{\alpha_P}{\sigma_P} = 2 \lambda \cdot \sigma_Q \cdot \rho_{PQ} = IR_Q \cdot \rho_{PQ}$$

The best we can do when chosing $\sigma_P$ for $P$ from above, is to set:

$$ \sigma_P = \sigma_Q \cdot \rho_{PQ} = \sigma_Q \cdot TC_P $$

Therefore, the improved fundamental law of active management is:

$$ IR_P = IC \cdot \sqrt{BR} \cdot TC_P $$

Another result from this is that, the **value-add** can be described as:

$$ VA_P \leq TC_P^2 \cdot VA_Q $$

The interpretation is that value-add for a practical portfolio with constraints
is related to the square of transfer coefficient. If $TC_P = 0.7$, we'd lose
nearly 50% of the value-add of the optimal portfolio, before other costs & fees!

Constraints can have a large impact on IR. Books examined the long-only constraint.
Results showed that with such constraints, IR frontier flattens out, i.e. for each additional unit of risk, expected alpha increases less and less.

Another side effect in a long-only portfolio is that, in a benchmark, for assets
with smaller weights, underweight is more constrainted vs assets with larger
weights.

The conclusion is that we are better off running long-only portfolios with low
residual risk, and use long-short implementations when we wish to run higher
residual risk portfolios.

Some research showed that by relaxing the long-only constraint just slightly,
this can improve the transfer coefficient significantly. Similar to the 80/20 rule,
where the first 20% shorts allowed, provided 80% of the benefits.


## Dynamic Portfolio Management

### Notation

- $IR_Q$ - potential information ratio
- $\lambda$ - risk penalty
- $HLY$ - half life in years
- $\chi$ - level of transaction cost

- $\alpha = {\alpha_1, \alpha_2, \cdots, \alpha_N}$ - forecast of excess return, $N \times 1$
- $p$ position weights for portfolio $P$
- $\alpha_P = \alpha^T \cdot p$ - expected alpha of portfolio $P$
- $\sigma_P$ or $\omega_P$ - risk of portfolio $P$, $\sigma_P = \sqrt{p^T \cdot V \cdot p}$
- $IR_P = \frac{\alpha_P}{\omega_P}$ - portfolio $P$ information ratio
- $c_P$ - expected annual -cost of $P$
- $O_P = {\alpha_P - c_P} - \frac{\lambda}{2}\sigma_P^2$  - the objective value of portfolio $P$
- $ACIR_P = \frac{\alpha_P - c_P}{\sigma_P}$ - the after cost information ratio of portfolio $P$

- Zeor-Cost portfolio $Q$ - no transaction portfolio with holdings $q(t)$, or just $q$
- $d$ - trade rate
- Target portfolio $M$ - with t-cost, has positions $m(t)$

- $N$ - no. of assets
- $V$ - covariance matrix for $N$ assets, $N \times N$
- **Uncertainty equivalent alpha** for $P$: $U_P = \alpha_P - \frac{\lambda}{2} \cdot \omega_P^2$, i.e. return - expected risk.

To maximise $U_P$ w.r.t. position size $p$, we have:

$$
\begin{aligned}
U_P &=  \alpha^T p - \frac{\lambda}{2} \cdot p^T \cdot V \cdot p \\
\frac{\partial{U_P}} {\partial{p}} &=  \alpha^T - \lambda \cdot V \cdot p \\
\therefore & \alpha - \lambda \cdot V \cdot p = 0 \\
\alpha &= \lambda \cdot V \cdot p & N \times 1
\end{aligned}
$$

For optimal portfolio $Q$ with position $q$, we have $\alpha = \lambda \cdot V \cdot q$.


### Signal Weighting
