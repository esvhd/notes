# Rethinking Model Examples

library(rethinking)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(logical = FALSE))

# Following section compares different modelling techniques for UCB admissions
# data.

data("UCBadmit")
d <- UCBadmit
d$dept_id <- coerce_index(d$dept)
d$male <- ifelse(d$applicant.gender=='male', 1, 0)

df <- data.frame(admit=d$admit,
                 dept_id=d$dept_id,
                 male=d$male,
                 applications=d$applications)

# Simple model where variation in departments are not taken into account

m1 <- map(
  alist(
    # binominal distributions, aplications is # of trials,
    # p is the number of positive examples.
    admit ~ dbinom(applications, p),
    logit(p) <- a + bm * male,
    a ~ dnorm(0, 10),
    bm ~ dnorm(0, 10)
  ),
  data=d
)
postcheck(m1)

# Allow different intercept for each department.
m2 <- map2stan(
  alist(
    admit ~ dbinom(applications, p),
    logit(p) <- a[dept_id] + bm * male,
    a[dept_id] ~ dnorm(0, 10),
    bm ~ dnorm(0, 10)
  ),
  data=df, chains=4, cores=4
)
precis(m2, depth=2)
postcheck(m2)

# Varying Intercept
m3 <- map2stan(
  alist(
    admit ~ dbinom(applications, p),
    logit(p) <- a[dept_id] + bm * male,
    a[dept_id] ~ dnorm(ax, sigma),
    ax ~ dnorm(0, 10),
    sigma ~ dcauchy(0, 1),
    bm ~ dnorm(0, 10)
  ),
  data=df, chains=4, cores=4
)
precis(m3, depth=3)
postcheck(m3)
plot(coeftab(m2, m3))

# Varying Intercept, extracting group average
m4 <- map2stan(
  alist(
    admit ~ dbinom(applications, p),
    logit(p) <- ga + a[dept_id] + bm * male,
    a[dept_id] ~ dnorm(0, sigma),
    ga ~ dnorm(0, 10),
    sigma ~ dcauchy(0, 1),
    bm ~ dnorm(0, 10)
  ),
  data=df, chains=4, cores=4, iter=5000, warmup=2000
)
precis(m4, depth=3)
postcheck(m4)
plot(coeftab(m2,m3,m4))

compare(m2, m3, m4)

# Varying slope & intercept. models correlation

m5 <- map2stan(
  alist(
    admit ~ dbinom(applications, p),
    logit(p) <- a[dept_id] + bm[dept_id] * male,
    c(a, bm)[dept_id] ~ dmvnorm2(c(ax, bmx), sigma_dept, Rho),
    ax ~ dnorm(0, 10),
    bmx ~ dnorm(0, 10),
    sigma_dept ~ dcauchy(0, 1),
    # LKJcorr defines a weakly informative prior on rho
    # think of it as a regularizing prior for correlations.
    Rho ~ dlkjcorr(2)
  ),
  data=df, chains=4, cores=4, iter=5000, warmup=2000
)
precis(m5, depth=2)
plot(precis(m5, pars=c('a', 'bm'), depth=2))
postcheck(m5)

compare(m2, m3, m4, m5)
