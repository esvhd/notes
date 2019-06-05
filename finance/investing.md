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
