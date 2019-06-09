# 积少成多

# Common Factors in Corporate Bond Returns

**Four factors: carry, quality, momentum and value.**

With long-short portfolio and cross-sectional regressions, the paper finds positive risk premium for all 3 factor but carry.

Uses **ex-ante** meansure of beta: Duration Times Spread (DTS).

Examined returns ~ 4 factors / combination of factors / macro factors, finds the factors have hedging effect, i.e. return of combined portfolio are higher when growth expectations are lower, volatility increases and inflation expectations increase.

Did not find that **broker-dealer leverage** can explain credit characteristic returns.

Chordia et al. 2016 looked at factors that explain equity excess returns for bonds. But equity and credit risks are not identical, the paper only documents correlation. 

Jostova et al 2013 examined credit momentum to invest in HY.

Frazzini Pedersen 2014 finds positive risk adjusted return to go long short duration higher rated paper vs short long duration lower rated paper.
NG and Phelps 2014 finds this is sensitive to selected measure of risk.

## Data

BAML monthly index return data. Filtering on the following:

* seniority - limits to only senior debt, most prevalent rating of the issuer.
* maturity - take 5yr-15yr bonds only if exist, otherwise keep all bonds
* age - use bonds > 2yrs old if possible and remove all others. Otherwise keep all.
* size - for the remaining bonds, pick largest outstanding.

48% of this universe can be found in Compustat database (financial statements).

Use S&P credit rating.

**Age percent** = time since issuance / original tenor

## Factors

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

## Portfolio Construction

Factors are demeaned within 5 ex-ante beta quintiles with beta measured as DTS, excluding duration and carry. 

Long short portfolio: Combine quintiles with inverse of risk weighted sum of all 4 factors. Weights are linear in rank (Asness et al, 2014). 

### Long Only Portfolio

Use equally weighted combination of 4 factors. Objective is to maximize portfolio weighted combined factor score, subject to constraints:

* no short
* max 25bp deviation from bench weight
* fully invested
* max 10% turnover
* min trade size $100k
* max deviation from bench spread 0.5%
* max duration deviation from bench 0.5
* rebalanced monthly.


# Defensive Factor Timing

Kristin Fergis, Katelyn Gallagher, Philip Hodges, Ked Hogans - BlackRock, 2019

## Risk Features

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

## Macro Factors

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
    - Equity: Shiller earnings yield
    - Vol: 
        + carry implied by the VIX term structure
        + ratio of current prices, spot VIX, to short-term realized vol of SPX index




