library(rethinking)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(logical = FALSE))

# Chapter 13 Examples, Cafe waiting times
a <- 3.5
b <- -1
sigma_a <- 1
sigma_b <- .5
# true correlation between a & b
rho <- -0.7

Mu <- c(a, b)
sigmas <- c(sigma_a, sigma_b)
Rho <- matrix(c(1, rho, rho, 1), nrow=2)
Sigma <- diag(sigmas) %*% Rho %*% diag(sigmas)

N_cafes <- 20

library(MASS)
set.seed(5)

vary_effects <- mvrnorm(N_cafes, Mu, Sigma)

a_cafe <- vary_effects[,1]
b_cafe <- vary_effects[,2]

plot(a_cafe, b_cafe, col=rangi2, 
     xlab='intercepts (a_cafe)', ylab='slopes (b_cafe)')

library(ellipse)
for (l in c(.1, .3, .5, .8, .99)) {
  lines(ellipse(Sigma, centre=Mu, level=l), col=col.alpha('black', .2))
}

N_visits <- 10
afternoon <- rep(0:1, N_visits * N_cafes / 2)
cafe_id <- rep(1:N_cafes, each=N_visits)

mu <- a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
sigma <- .5
wait <- rnorm(N_visits * N_cafes, mu, sigma)

d <- data.frame(cafe=cafe_id, afternoon=afternoon, wait=wait)

m13.1 <- map2stan(
  alist(
    wait ~ dnorm(mu, sigma),
    mu <- a_cafe[cafe] + b_cafe[cafe] * afternoon,
    c(a_cafe, b_cafe)[cafe] ~ dmvnorm2(c(a, b), sigma_cafe, Rho),
    a ~ dnorm(0, 10),
    b ~ dnorm(0, 10),
    sigma_cafe ~ dcauchy(0, 2),
    sigma ~ dcauchy(0, 2),
    Rho ~ dlkjcorr(2)
  ),
  data=d,
  iter=6e3, warmup=2e3, chains=3, cores=3
)

precis(m13.1, depth=2)

# LKJcorr(2)
post <- extract.samples(m13.1)
dens(post$Rho[,1,2], col='red', lwd=2)
R <- rlkjcorr(1e5, K=2, eta=2)
dens(R[,1,2], add=TRUE, lty=2, col='darkgreen', xlab='correlation')

# LKJcorr(5)
post <- extract.samples(m13.2)
dens(post$Rho[,1,2], col='blue', lwd=2, add=TRUE)
R2 <- rlkjcorr(1e5, K=2, eta=5)
dens(R2[,1,2], add=TRUE, lty=2, col='black')

# LKJcorr(5)
post <- extract.samples(m13.3)
dens(post$Rho[,1,2], col='cyan', lwd=2, add=FALSE)
R3 <- rlkjcorr(1e5, K=2, eta=1)
dens(R3[,1,2], add=TRUE, lty=4, col='black')

# LKJcorr(1)
post <- extract.samples(m13.1)
dens(post$Rho[,1,2], col='red', lwd=2)
R <- rlkjcorr(1e5, K=2, eta=2)
dens(R[,1,2], add=TRUE, lty=2, col='darkgreen')

post <- extract.samples(m13.3)
dens(post$Rho[,1,2], col='cyan', lwd=2, add=TRUE)


## R code 13.14
# compute unpooled estimates directly from data
a1 <- sapply( 1:N_cafes ,
              function(i) mean(wait[cafe_id==i & afternoon==0]) )
b1 <- sapply( 1:N_cafes ,
              function(i) mean(wait[cafe_id==i & afternoon==1]) ) - a1

# extract posterior means of partially pooled estimates
post <- extract.samples(m13.1)
a2 <- apply( post$a_cafe , 2 , mean )
b2 <- apply( post$b_cafe , 2 , mean )

# plot both and connect with lines
plot( a1 , b1 , xlab="intercept" , ylab="slope" ,
      pch=16 , col=rangi2 , ylim=c( min(b1)-0.1 , max(b1)+0.1 ) ,
      xlim=c( min(a1)-0.1 , max(a1)+0.1 ) )
points( a2 , b2 , pch=1 )
for ( i in 1:N_cafes ) lines( c(a1[i],a2[i]) , c(b1[i],b2[i]) )

## R code 13.15
# compute posterior mean bivariate Gaussian
Mu_est <- c( mean(post$a) , mean(post$b) )
rho_est <- mean( post$Rho[,1,2] )
sa_est <- mean( post$sigma_cafe[,1] )
sb_est <- mean( post$sigma_cafe[,2] )
cov_ab <- sa_est*sb_est*rho_est
Sigma_est <- matrix( c(sa_est^2,cov_ab,cov_ab,sb_est^2) , ncol=2 )

# draw contours
library(ellipse)
for ( l in c(0.1,0.3,0.5,0.8,0.99) )
  lines(ellipse(Sigma_est,centre=Mu_est,level=l),
        col=col.alpha("black",0.2))

## R code 13.16
# convert varying effects to waiting times
wait_morning_1 <- (a1)
wait_afternoon_1 <- (a1 + b1)
wait_morning_2 <- (a2)
wait_afternoon_2 <- (a2 + b2)

# plot both and connect with lines
plot( wait_morning_1 , wait_afternoon_1 , 
      xlab="morning wait" , ylab="afternoon wait" ,
      pch=16 , col=rangi2 , 
      ylim=c( min(wait_afternoon_1)-0.1 , max(wait_afternoon_1)+0.1 ) ,
      xlim=c( min(wait_morning_1)-0.1 , max(wait_morning_1)+0.1 ) )
points( wait_morning_2 , wait_afternoon_2 , pch=1 )
for ( i in 1:N_cafes ) lines( c(wait_morning_1[i],wait_morning_2[i]) , 
                              c(wait_afternoon_1[i],wait_afternoon_2[i]) )

Mu_est <- c(mean(wait_morning_2), mean(wait_afternoon_2))
sd_am_2 <- sd(wait_morning_2)
sd_pm_2 <- sd(wait_afternoon_2)
cov_est <- cov(wait_morning_2, wait_afternoon_2)
Sigma_est <- matrix(c(sd_am_2^2, cov_est, cov_est, sd_pm_2^2), ncol=2)

# draw contours
library(ellipse)
for ( l in c(0.1,0.3,0.5,0.8,0.99) )
  lines(ellipse(Sigma_est,centre=Mu_est,level=l),
        col=col.alpha("black",0.2))

lines(1:5, 1:5, lty=2)


# 13M2

m13.m2 <- map2stan(
  alist(
    wait ~ dnorm(mu, sigma),
    mu <- a_cafe[cafe] + b_cafe[cafe] * afternoon,
    a_cafe[cafe] ~ dnorm(a, a_sigma),
    b_cafe[cafe] ~ dnorm(b, b_sigma),
    a ~ dnorm(0, 10),
    b ~ dnorm(0, 10),
    a_sigma ~ dcauchy(0, 1),
    b_sigma ~ dcauchy(0, 1),
    sigma ~ dcauchy(0, 1)
  ),
  data=d,
  iter=6e3, warmup=2e3, chains=3, cores=3
)
precis(m13.m2, depth=2)

compare(m13.1, m13.m2)
plot(compare(m13.1, m13.m2))


# 13M3

data("UCBadmit")
d <- UCBadmit
d$male <- ifelse(d$applicant.gender == 'male', 1, 0)
d$dept_id <- coerce_index(d$dept)

m13.3 <- map2stan(
  alist(
    admit ~ dbinom( applications , p ),
    logit(p) <- a_dept[dept_id] +
      bm_dept[dept_id]*male,
    c(a_dept,bm_dept)[dept_id] ~ dmvnorm2( c(a,bm) , sigma_dept , Rho ),
    a ~ dnorm(0,10),
    bm ~ dnorm(0,1),
    sigma_dept ~ dcauchy(0,2),
    Rho ~ dlkjcorr(2)
  ),
  data=d , warmup=1000 , iter=5000 , chains=4 , cores=3 )

m13.m3 <- map2stan(
  alist(
    admit ~ dbinom( applications , p ),
    logit(p) <- A + a_dept[dept_id] + bm_dept[dept_id]*male,
    c(a_dept,bm_dept)[dept_id] ~ dmvnormNC(sigma_dept , Rho ),
    A ~ dnorm(0, 10),
    sigma_dept ~ dcauchy(0,2),
    Rho ~ dlkjcorr(2)
  ),
  data=d , warmup=1000 , iter=5000 , chains=4 , cores=3 )

m13M3 <- map2stan(
  alist(
    admit ~ dbinom( applications , p ),
    logit(p) <- A + a_dept[dept_id] + BM + bm_dept[dept_id]*male,
    c(a_dept,bm_dept)[dept_id] ~ dmvnormNC(sigma_dept , Rho ),
    A ~ dnorm(0, 10),
    BM ~ dnorm(0, 10),
    sigma_dept ~ dcauchy(0,2),
    Rho ~ dlkjcorr(2)
  ),
  data=d , warmup=1000 , iter=5000 , chains=4 , cores=3 )


par_list <- c('a','bm','a_dept','b_dept','sigma_dept','Rho')
n_eff1 <- precis(m13.3,2,pars=par_list)@output[,'n_eff']
n_eff2 <- precis(m13.m3,2,pars=par_list)@output[,'n_eff']
n_eff3 <- precis(m13M3,2,pars=par_list)@output[,'n_eff']
cbind( n_eff1 , n_eff2, n_eff3 )


# answer for 13M3 - two separate intercepts, wouldn't this cause problems
# for the model??
post1 <- extract.samples(m13.m3)
post2 <- extract.samples(m13M3)

cc <- post2$A + post2$BM

# density show similar results thought. HMC is quite good at that.
dens(post1$A)
dens(cc, add=TRUE, lty=2)

# 13H1
data('bangladesh')
d <- bangladesh
str(d)

df <- data.frame(use_contra=d$use.contraception, 
                 district_id=coerce_index(d$district),
                 urban=d$urban) 

m13H1 <- map2stan(
  alist(
    use_contra ~ dbinom(1, p),
    logit(p) ~ A + a_dist[district_id] + (BM + bm[district_id]) * urban,
    c(A, BM) ~ dnorm(0, 10),
    c(a_dist, bm)[district_id] ~ dmvnorm2(0, sigma, Rho),
    sigma ~ dcauchy(0, 1),
    Rho ~ dlkjcorr(2)
  ),
  data=df, chains=3, cores=3, iter = 5e3, warmup = 1e3
)

post <- extract.samples(m13H1)
dens(post$Rho[,1,2])


pred.urban <- data.frame(urban=rep(1, 60), district_id=seq(1:60))
pred.rural <- data.frame(urban=rep(0, 60), district_id=seq(1:60))

urban.fit <- link(m13H1, data=pred.urban)
rural.fit <- link(m13H1, data=pred.rural)

urban.mean <- apply(urban.fit, 2, mean)
rural.mean <- apply(rural.fit, 2, mean)

plot(rural.mean, urban.mean, xlab='Rural', ylab='Urban', col='slateblue',
     xlim=c(0, 1), ylim=c(0, 1))
abline(a=0, b=1, lty=2)


plot(rural.mean, urban.mean-rural.mean, 
     xlab='Rural', ylab='Urban - Rural', col='slateblue',
     xlim=c(0, 1), ylim=c(-.5, .5))
abline(h=0, lty=2)


# 13H2

data('Oxboys')
d <- Oxboys
str(d)

# I prefer this model as it constrains that mu is a positive number. 
# Although the distrutional choice is still not a great one. Perhaps try
# gamma distribution?
# d$height_log <- log(d$height)
m2 <- map2stan(
  alist(
    height ~ dnorm(mu, sigma),
    log(mu) <- A + a[Subject] + (BA + b[Subject]) * age,
    A ~ dnorm(0, 100),
    BA ~ dnorm(0, 10),
    c(a, b)[Subject] ~ dmvnorm2(0, sigma_ab, Rho),
    sigma_ab ~ dcauchy(0, 2),
    sigma ~ dcauchy(0, 2),
    Rho ~ dlkjcorr(2)
  ),
  data=d, iter=6e3, warmup = 2e3, chains = 3, cores = 3
)

precis(m2, depth=2)

m3 <- map2stan(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- A + a[Subject] + (BA + b[Subject]) * age,
    A ~ dnorm(0, 100),
    BA ~ dnorm(0, 10),
    c(a, b)[Subject] ~ dmvnorm2(0, sigma_ab, Rho),
    sigma_ab ~ dcauchy(0, 2),
    sigma ~ dcauchy(0, 2),
    Rho ~ dlkjcorr(2)
  ),
  data=d, iter=6e3, warmup = 2e3, chains = 3, cores = 3
)
precis(m3, depth=2)

m4 <- map2stan(
  alist(
    height_log ~ dlnorm(mu, sigma),
    mu <- A + a[Subject] + (BA + b[Subject]) * age,
    A ~ dnorm(0, 100),
    BA ~ dnorm(0, 10),
    c(a, b)[Subject] ~ dmvnorm2(0, sigma_ab, Rho),
    sigma_ab ~ dcauchy(0, 2),
    sigma ~ dcauchy(0, 2),
    Rho ~ dlkjcorr(2)
  ),
  data=d, iter=6e3, warmup = 2e3, chains = 3, cores = 3
)


# two models have different correlation. Make sense given one is log-normal.
# intercepts and slopes show positive correlation. Implies that for boys 
# with higher than average height, age is adds more to prediction.
post <- extract.samples(m2)
postx <- extract.samples(m3)
dens(post$Rho[,1,2], xlab='correlation', 
     ylim=c(0, 3.), xlim=c(-.3, 1.))
dens(postx$Rho[,1,2], lty=2, add=TRUE)
legend('topleft', c('log-normal, m2', 'normal, m3'), lty=c(1, 2))

# compare intercept A.
dens(exp(post$A), xlab='Grand Intercept')
dens(postx$A, add=T, lty=2)
legend('topleft', c('log-normal, m2', 'normal, m3'), lty=c(1, 2))


# 13H4

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Use m3 works, but m2 still does not.
post <- extract.samples(m2)
a <- mean(post$A)
b <- mean(post$BA)
sd_a <- mean(post$sigma_ab[,1])
sd_b <- mean(post$sigma_ab[,2])
rho <- Mode(post$Rho[,1,2])

# sigma for m2, approx. Don't know how to get it from model yet.
# use rlnorm?
sigma <- sd(post$A)
# sigma <- log(mean(post$sigma))

# sigma for m3 model
# sigma <- mean(post$sigma)

S <- matrix(c(sd_a^2, sd_a*sd_b*rho, sd_a*sd_b*rho, sd_b^2 ), nrow=2)

library(MASS)
set.seed(5)
sims <- mvrnorm(10, c(0, 0), Sigma = S)

age.seq <- seq(-1, 1, length.out = 10)

plot(0, 0, type='n', xlab='age', ylab='height', xlim=c(-1, 1), ylim=c(120, 170))
for (i in 1:nrow(sims)) {
  mean_sim <- a + sims[i,1] + (b + sims[i, 2]) * age.seq
  # if (mean_sim <= 0) {
  #   print(mean_sim)
  #   next
  # }
  # print(mean_sim)
  h <- rlnorm(10,
             meanlog=a + sims[i,1] + (b + sims[i, 2]) * age.seq,
             sdlog=sigma)
  print(h)
  # h <- rnorm(10,
  #                mean=a + sims[i,1] + (b + sims[i, 2]) * age.seq,
  #                sd=sigma)
  lines(age.seq, h, col=col.alpha('slateblue', .3))
}

# Chapter 14

# 14M2

data(milk)
d <- milk
dcc <- d[complete.cases(d),]
str(dcc)
dat <- data.frame(
  log_mass = log(d$mass),
  kcal = d$kcal.per.g,
  neocortex = d$neocortex.perc / 100
)


m14m2 <- map2stan(
  alist(
    kcal ~ dnorm(mu, sigma),
    mu <- a + bN * neocortex + bM * log_mass,
    neocortex ~ dnorm(nu, sigma_n),
    nu ~ dnorm(0, 10),
    c(bN, bM) ~ dnorm(0, 10),
    sigma ~ dcauchy(0, 2),
    sigma_n ~ dcauchy(0, 2),
    a ~ dnorm(0, 10)
  ),
  data=dat, chains=1, cores=1
)
precis(m14m2, depth=2, pars = c('a', 'bN', 'bM', 'sigma'))


m14m2x <- map2stan(
  alist(
    kcal ~ dnorm(mu, sigma),
    mu <- a + bN * neocortex + bM * log_mass,
    neocortex ~ dnorm(nu, sigma_n),
    nu <- aN + gM * log_mass,
    aN ~ dnorm(0, 10),
    gM ~ dnorm(0, 10),
    c(bN, bM) ~ dnorm(0, 10),
    sigma ~ dcauchy(0, 2),
    sigma_n ~ dcauchy(0, 2),
    a ~ dnorm(0, 10)
  ),
  data=dat, chains=3, cores=3, iter=5e3, warmup = 1e3
)
precis(m14m2x, depth=2, pars=c('a', 'bN', 'bM', 'aN', 'gM', 'sigma'))
plot(coeftab(m14m2, m14m2x)@coefs[c('a', 'bN', 'bM', 'aN', 'gM', 'sigma'),])

dnc <- dat[!complete.cases(dat$neocortex),]

post <- extract.samples(m14m2x)

plot(neocortex ~ log_mass, data=dat, pch=16, col='slateblue')
# imputed values are given by the model. but need to map it to log_mass


# 14M3

data(WaffleDivorce)
d <- WaffleDivorce
dlist <- list(
  div_obs=d$Divorce,
  div_sd=d$Divorce.SE * 2 , # note times 2
  R=d$Marriage,
  A=d$MedianAgeMarriage )

m14M3 <- map2stan(
  alist(
    div_est ~ dnorm(mu,sigma),
    mu <- a + bA*A + bR*R,
    div_obs ~ dnorm(div_est,div_sd),
    a ~ dnorm(0,10),
    bA ~ dnorm(0,10),
    bR ~ dnorm(0,10),
    sigma ~ dcauchy(0,2.5)
  ),
  data=dlist ,
  start=list(div_est=dlist$div_obs) ,
  WAIC=FALSE , iter=5000 , warmup=1000 , chains=3 , cores=3 , 
  control=list(adapt_delta=0.995) )

precis(m14M3)


# 14H1

data("elephants")
d <- elephants
str(d)

m1 <- map2stan(
  alist(
    MATINGS ~ dpois(lambda),
    log(lambda) <- a + b * AGE,
    a ~ dnorm(0, 10),
    b ~ dnorm(0, 1)
  ),
  data=d, chains=3, cores=3, iter=4000
)
precis(m1)

m2 <- map2stan(
  alist(
    MATINGS ~ dpois(lambda),
    log(lambda) <- a + b * AGE_est[i],
    AGE ~ dnorm(AGE_est, 5),
    a ~ dnorm(0, 10),
    b ~ dnorm(0, 10)
  ),
  data=d, 
  start=list(AGE_est=d$AGE),
  chains=3, cores=3, iter=4000, WAIC=F
)

precis(m2)

compare(m1, m2)

age.seq <- seq(from=20, to=60, by=1)
fit1 <- link(m1, data=list(AGE=age.seq))
lambda1 <- apply(fit1, 2, mean)
lambda1.pi <- apply(fit1, 2, PI)

plot(MATINGS ~ AGE, data=d, col='slateblue')
lines(age.seq, lambda1)
shade(lambda1.pi, age.seq)

post <- extract.samples(m2)
AGE_mu <- apply(post$AGE_est, 2, mean)
matings_j <- jitter(d$MATINGS)

# plots show that model fitted values gravitate towrads regression line, in 
# terms of age.
plot(d$AGE, matings_j, col=rangi2, xlab='age', ylab='matings',
     xlim=c(20, 60))
lines(age.seq, lambda1)
points(AGE_mu, matings_j)
for (i in 1:nrow(d))
  lines(c(d$AGE[i], AGE_mu[i]), rep(matings_j[i], 2))


m3 <- map2stan(
  alist(
    MATINGS ~ dpois(lambda),
    log(lambda) <- a + b * AGE_est[i],
    AGE ~ dnorm(AGE_est, 100),
    a ~ dnorm(0, 10),
    b ~ dnorm(0, 10)
  ),
  data=d,
  start=list(AGE_est=d$AGE),
  control=list(adapt_delta=.99),
  chains=1, cores=1, iter=4000, WAIC=F, warmup = 2000
)
precis(m3, depth=2)
plot(m3, pars='b', window=c(2000, 4000), n_col=1)


# 14H3

set.seed(100)
x <- c( rnorm(10) , NA )
y <- c( rnorm(10,x) , 100 )
d <- list(x=x,y=y)

plot(y ~ x, data=d)

m4 <- map2stan(
  alist(
    y ~ dnorm(mu, sigma),
    mu <- a + b * x[i],
    x ~ dnorm(0, 10),
    c(a, b) ~ dnorm(0, 100),
    sigma ~ dcauchy(0, 1)
  ),
  data=d, WAIC=F, chains=3, cores=3, iter=5e3, warmup = 2e3,
  control=list(adapt_level=.99)
)


## R code 13.22
library(rethinking)
data(chimpanzees)
d <- chimpanzees
d$recipient <- NULL
d$block_id <- d$block

m13.6 <- map2stan(
  alist(
    # likeliood
    pulled_left ~ dbinom(1,p),
    
    # linear models
    logit(p) <- A + (BP + BPC*condition)*prosoc_left,
    A <- a + a_actor[actor] + a_block[block_id],
    BP <- bp + bp_actor[actor] + bp_block[block_id],
    BPC <- bpc + bpc_actor[actor] + bpc_block[block_id],
    
    # adaptive priors
    c(a_actor,bp_actor,bpc_actor)[actor] ~
      dmvnorm2(0,sigma_actor,Rho_actor),
    c(a_block,bp_block,bpc_block)[block_id] ~
      dmvnorm2(0,sigma_block,Rho_block),
    
    # fixed priors
    c(a,bp,bpc) ~ dnorm(0,1),
    sigma_actor ~ dcauchy(0,2),
    sigma_block ~ dcauchy(0,2),
    Rho_actor ~ dlkjcorr(4),
    Rho_block ~ dlkjcorr(4)
  ) , data=d , iter=5000 , warmup=1000 , chains=3 , cores=3 )

precis(m13.6, depth=2)

## R code 13.23
m13.6NC <- map2stan(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- A + (BP + BPC*condition)*prosoc_left,
    A <- a + a_actor[actor] + a_block[block_id],
    BP <- bp + bp_actor[actor] + bp_block[block_id],
    BPC <- bpc + bpc_actor[actor] + bpc_block[block_id],
    # adaptive NON-CENTERED priors
    c(a_actor,bp_actor,bpc_actor)[actor] ~
      dmvnormNC(sigma_actor,Rho_actor),
    c(a_block,bp_block,bpc_block)[block_id] ~
      dmvnormNC(sigma_block,Rho_block),
    c(a,bp,bpc) ~ dnorm(0,1),
    sigma_actor ~ dcauchy(0,2),
    sigma_block ~ dcauchy(0,2),
    Rho_actor ~ dlkjcorr(4),
    Rho_block ~ dlkjcorr(4)
  ) , data=d , iter=5000 , warmup=1000 , chains=3 , cores=3 )

m13.6NC2 <- map2stan(
  alist(
    pulled_left ~ dbinom(1,p),
    logit(p) <- A + (BP + BPC*condition)*prosoc_left,
    A <- a + a_actor[actor] + a_block[block_id],
    BP <- bp_actor[actor] + bp_block[block_id],
    BPC <- bpc_actor[actor] + bpc_block[block_id],
    # adaptive NON-CENTERED priors
    c(a_actor,bp_actor,bpc_actor)[actor] ~
      dmvnormNC(sigma_actor,Rho_actor),
    c(a_block,bp_block,bpc_block)[block_id] ~
      dmvnormNC(sigma_block,Rho_block),
    a ~ dnorm(0,1),
    sigma_actor ~ dcauchy(0,2),
    sigma_block ~ dcauchy(0,2),
    Rho_actor ~ dlkjcorr(4),
    Rho_block ~ dlkjcorr(4)
  ) , data=d , iter=5000 , warmup=1000 , chains=3 , cores=3 )

## R code 13.24
# extract n_eff values for each model
neff_c <- precis(m13.6,2)@output$n_eff
neff_nc <- precis(m13.6NC,2)@output$n_eff
neff_nc2 <- precis(m13.6NC2,2)@output$n_eff
# plot distributions
boxplot( list( 'm13.6'=neff_c , 'm13.6NC'=neff_nc,  'm13.6NC2'=neff_nc2) ,
         ylab="effective samples" , xlab="model" )

## R code 13.25
precis( m13.6NC , depth=2 , pars=c("sigma_actor","sigma_block") )

precis(m13.6NC, depth=2)
precis(m13.6NC2, depth=2)

compare(m13.6, m13.6NC, m13.6NC2)

## R code 13.26
p <- link(m13.6NC)
str(p)

## R code 13.27
compare( m13.6NC , m12.5 )

## R code 13.28
m13.6nc1 <- map2stan(
  alist(
    pulled_left ~ dbinom(1,p),
    
    # linear models
    logit(p) <- A + (BP + BPC*condition)*prosoc_left,
    A <- a + za_actor[actor]*sigma_actor[1] +
      za_block[block_id]*sigma_block[1],
    BP <- bp + zbp_actor[actor]*sigma_actor[2] +
      zbp_block[block_id]*sigma_block[2],
    BPC <- bpc + zbpc_actor[actor]*sigma_actor[3] +
      zbpc_block[block_id]*sigma_block[3],
    
    # adaptive priors
    c(za_actor,zbp_actor,zbpc_actor)[actor] ~ dmvnorm(0,Rho_actor),
    c(za_block,zbp_block,zbpc_block)[block_id] ~ dmvnorm(0,Rho_block),
    
    # fixed priors
    c(a,bp,bpc) ~ dnorm(0,1),
    sigma_actor ~ dcauchy(0,2),
    sigma_block ~ dcauchy(0,2),
    Rho_actor ~ dlkjcorr(4),
    Rho_block ~ dlkjcorr(4)
  ) ,
  data=d ,
  start=list( sigma_actor=c(1,1,1), sigma_block=c(1,1,1) ),
  constraints=list( sigma_actor="lower=0", sigma_block="lower=0" ),
  types=list( Rho_actor="corr_matrix", Rho_block="corr_matrix" ),
  iter=5000 , warmup=1000 , chains=3 , cores=3 )
