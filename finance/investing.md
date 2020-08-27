# 积少成多

## Common Factors in Corporate Bond Returns

**Four factors: carry, quality, momentum and value.**

With long-short portfolio and cross-sectional regressions, the paper finds positive risk premium for all 3 factor but carry.

Uses **ex-ante** meansure of beta: Duration Times Spread (DTS).

Examined returns ~ 4 factors / combination of factors / macro factors, finds the factors have hedging effect, i.e. return of combined portfolio are higher when growth expectations are lower, volatility increases and inflation expectations increase.

Did not find that **broker-dealer leverage** can explain credit characteristic returns.

Chordia et al. 2016 looked at factors that explain equity excess returns for bonds. But equity and credit risks are not identical, the paper only documents correlation.

Jostova et al 2013 examined credit momentum to invest in HY.

Frazzini Pedersen 2014 finds positive risk adjusted return to go long short duration higher rated paper vs short long duration lower rated paper.
NG and Phelps 2014 finds this is sensitive to selected measure of risk.

### Data

BAML monthly index return data. Filtering on the following:

* seniority - limits to only senior debt, most prevalent rating of the issuer.
* maturity - take 5yr-15yr bonds only if exist, otherwise keep all bonds
* age - use bonds > 2yrs old if possible and remove all others. Otherwise keep all.
* size - for the remaining bonds, pick largest outstanding.

48% of this universe can be found in Compustat database (financial statements).

Use S&P credit rating.

**Age percent** = time since issuance / original tenor

### Factors

More detail in appendix Table A.1 in paper.

**Carry** = OAS, not yield given interest in excess returns.

OAS problems: it represents carry IFF the credit curve is flat. But it's transparent and simple.

**Defensive**:

* Market leverage = net debt / (net debt + market value of equity), where net debt = (book debt + minority interest + preferred stocks - cash).
* Gross profitability defined in Novy-Marx 2013.
* Effective Duration (lower is more defensive)
* Excluding: beta and vol (somewhat captured by DTS)

Evidence of netaive relationship between beta and future excess returns in cash markets. (Frazzini and Pedersen 2014)

**Momentum**:

* trailing 6-month bond cumulative excess return
* 6-month cumulative equity return of the bond issuer

**Value**:

Spread / default risk - cheap bond has high spread relative to default risk. Not very clear how this is done.

* issuer default probability derived from public equity (Bharath and Shumway 2008)
* combination of credit rating, bond duration and vol of last 12 months excess returns

In more precise terms:

* residual from  `log(OAS) ~ log(duration) + rating + 12-month_excess_return_vol`
* residual from `log(OAS) ~ log(default_prob)`

### Portfolio Construction

Factors are demeaned within 5 ex-ante beta quintiles with beta measured as DTS, excluding duration and carry.

Long short portfolio: Combine quintiles with inverse of risk weighted sum of all 4 factors. Weights are linear in rank (Asness et al, 2014).

#### Long Only Portfolio

Use equally weighted combination of 4 factors. Objective is to maximize portfolio weighted combined factor score, subject to constraints:

* no short
* max 25bp deviation from bench weight
* fully invested
* max 10% turnover
* min trade size $100k
* max deviation from bench spread 0.5%
* max duration deviation from bench 0.5
* rebalanced monthly.


## Defensive Factor Timing

Kristin Fergis, Katelyn Gallagher, Philip Hodges, Ked Hogans - BlackRock, 2019

### Risk Features

Use macro indicators and market measures to develope two features:

* Risk Tolerance Index (RTI) = $corr[q(R_t), q(\sigma_t)]$
    - $q(R^i_t)$ denotes the ranking of asset $i$ for time period $t$ - the paper used **3-month rolling periods with weekly returns**.
    - $q(\sigma^i_t)$ denotes the ranking of vol for the same assets
    - Essentially this measures the correlation of two ranking vectors, returns and vol.
* Diversification Ratio (DR) = $\frac{\sum_i w_i \sigma_i}{\sigma_p}$, where:
    - $w_i$ is the weight for asset $i$
    - $\sigma_i$ is the vol of asset $i$
    - $\sigma_p$ is the vol of the portfolio

The paper used these two indicators to de-risk a macro-factor portfolio.

### Macro Factors

Representative factors, not reflecting actual investment accounts. Portfolio allocates to each factor based on % of risk / vol budget.

The problem with this model here I see is that looking at the valuation metrics for these factors, there isn't much variation over time - they can be either rich or cheap for very long periods of time.

**Economic Growth**:

* Long: equity futures, listed REITS, commodities
* Short: cash
* Valuation Indcator:
    - variation of Shiller's CAPE: market-value weighted global CAPE then take inverse to compute earnings yield.
    - Difference between earnings yield and bond yields

**Real Rates**:

* Long: baset of sovereign inflation linkers
* Short: cash
* Indcators: real rates vs expected GDP growth

**Inflation**:

* Long: basket of nominal sovereign bonds
* Short: basket of sovereign inflation linkers with matching maturity
* Indcators: 5yr market inflation breakeven vs inflation expectation (surveys and trailing realized inflation)

**Credit**:

* Long: IG and HY bonds
* Short: Govt bonds
* Indcators: current spreads vs loss given default

**EM**:

* Long: EM equity, EM debts
* Short: DM equity, DM govt bonds
* Indcators:
    - Equity: Shiller's earnings yield and dividend yield (not sure why div yield is needed...)
    - Bonds: EM spread over UST.
    - Political risk regression model

**Liqiudity**:

* Long: Small-cap equity
* Short: Large-cap equity, vol futures
* Indicators:
  * Equity: Shiller earnings yield
  * Vol:
      + carry implied by the VIX term structure
      + ratio of current prices, spot VIX, to short-term realized vol of SPX index


## Rethinking Alternative Data in Institutional Investment

Describes a **defensive and defensible** strategy towards using alternative data.

3 V's of big data + IBM's new V:

* Volume
* Velocity
* Variety
* Veracity - the degree of uncertainty around a dataset

Authors proposed 6 attributes to access alt-data:

* Reliability
* Granularity
* Freshness
* Comprehensiveness
* Actionability
* Scarcity

Interesting point that if investors are paying 3rd parties for their alt-data capability, in the long run this is just subsidising the 3rd party in enhancing their capability.

## Factors

### Beta Against Beta (BAB)

Aslo see related topic from [Low-Risk Anomalies in Global Fixed Income: Evidence from Major Broad Markets, Carvalho et al., 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2321012)

## Forecasting US HY Default / Portfolio construction

Barclays published a report on 2020-01-02 (Jay Hyman and co) on a 5-factor logistic regression model to forecast bond default probabilities. The output is then used to aid portfolio construction.

Portfolios are separated into 5 buckets, Q1 (worst) to Q5 (best), using two scoring methods:

1. raw predicted default probabilities
2. spread / predicted default probabilities

Method 2 produced better result in a `Q5 - Q1` portfolio.

The 5 factors are:

1. Cross-sectionally normalised OAS levels
2. VIX
3. tbd
4. Short term leverage (1yr debt maturities / equity MV)
5. past 6-months equity momentum

Observations:

* OAS outright level isn't used. Since when yields reach high levels such as 10%+ many names seem to reach the point of no return. Also default waves only come in high yielding environments.
  * Since this may be correlated with VIX and 6m equity performance, it might not be useful in this model setup.
* What other fundamental metrics were tested?
  * FCF / debt?
  * ROE?


### AQR Is Systematic Value Dead?

Date: 2020-05-08

Arguments against relevance of price to book ratio as a value measure in today's environment (2020):

* Strong firms today have higher intangile assets than the past, which are not captured in price to book.
* The best firms have more monopoly power today than in the past (winner takes all).
* Index / passive funds pushing valuation up further for higher market cap stocks, therefore skewing the value spread measure.

Additional measures to consider:

* price to sales (author claims this is far more relevant when we remove industry bet)
* 1yr trailing P/E ratio
* 1yr forward P/E ratio

Make these measures industry neutral - i.e. measuring within the industry and then take the average.

Article used equal weight method to construct portfolio, stayed with the largest 1000 stocks for this reason.

## Credit

### Banks

This is 2020, in the middle of the covid-19 pandemic crisis. I ran a simple model to access relative value in Euro LT2
bank bonds. I used BAML data, so rating methodology was to use the lowest rating.

A few relationships tested here are:

`oas ~ roe + tier1_capital + effective_duration + equity_6m_vol + rating_bucket` - to my surprise, bank tier 1 capital has a positive coefficient. That was counter-intuitive. What it means I think is that market did not care about capital - because almost all banks had tons of capital. It was not what drove market. Plus we all know that with the difficulties the economy is facing, in the area of 10-15% GDP hits in many developed countries, bank capital would need to take a hit. **Return of equity**, however, had a negative coefficient, meaning the higher RoE is, the lower the spreads. That is intuitive, suggesting that market cares more about the pace in which capital can be rebuilt. We are not pricing a doomsday's environment, investors believed that banks can get through this.

I also tried `oas ~ roe + tier1_capital + duration + rating_bucket`, `oas ~ roe + duration + equity_6m_vol`,
and `oas ~ roe + duration + equity_6m_vol + rating_bucket`.

Vol also did not matter, but rating buckets did. This made sense because in a non-doomsday world, investors are trying to predict where each layers of capital would land in the rating spectrum. That is the game right now.

If this is the thesis, what I need is more robust validation.

Effective duration - I'm not sure it truely represents the risk for LT2. Perhaps the best measure of potential extension risk is simply (maturity - next_call) measured in years.

Question that should be asked:

1. Is the market right in thinking that capital level isn't as important?
2. Where do we forecast RoE to be for each bank?
3. Where do we think ratings would land for each capital structure.

This would help us to use the model to access where do we think spreads would end up and help us make relative choices.

Validation strategy.

1. Identify factors that could drive returns
2. Run regression during different periods, compare the results.
3. Use time series CV for model and feature selection, train on a period's data, have a 1 week embargo, assess MSE or MAE for the next period.

## Recession Forecast

I've seen many of these research paper coming out from sell side houses. All of these publications show some sort of recession probablity chart over a super long horizon, e.g. 50 years. The problem is, a chart like this is bound to look good. But you need more details than a good looking chart. Specifically, test set performance evaluation results for with metrics suitable for imbalanced data. I rarely see these paper publish such results.

Some papers on this stuff.

### ML for Recession Prediction and Dynamic AA

[paper](https://jfds.pm-research.com/content/1/3/41)

This paper only uses classification error as a metric for evaluating a SVM model.

Features:

1. monthly log diff in non-farm payroll
2. log diff in average monthly SPX
3. ISM Manufacturing Production index.
4. 10-year UST yield - Fed Funds rate
5. 11 lags for all of the above 4, ie. 1yr's data, 48 total features.

Data revision is only visible after the date it's made.

### ML Prediction of Recessions

[link](https://jfds.pm-research.com/content/early/2020/08/21/jfds.2020.1.040)

This one at least has some proper ML metrics for evaluation.

NBER recession dataset from 1959 to 2019. 732 monthly data points, of which 101 (~14%) are flagged as recessions. So this indicates that a naive strategy of prediction no-recession all the time would have ~14% classification error.

Used down-sampling to handle imbalanced data.

Features:

<img src='img/yazdani_features.png' width=600/>

Performance:

<img src='img/yazdani.png' width=600/>
