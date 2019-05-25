# Reinforcement Learning Notes 

Based on David Silver's 2016 UCL course, videos [here](#https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

[Slides](http://goo.gl/vUiyjq)

# Notation

$\mathcal{S}$ - a finite set of **states**.

$\mathcal{P}$ - state **transition probability matrix**.

$\mathcal{R}$ - **reward function**.

$\gamma$ - **reward discount factor**.

$G_t$ - **Return**, total discounted reward from time step $t$. $G_t = \sum_{k=0}^{\infty}\gamma^k R_{t + k + 1}$.

$v(s)$ - **state-value function** for a Markove Reward Process is the **expected return** starting from state $s$.

$\mathcal{A}$ - a finite set of **actions**.

$\pi$ - **policy**

$v_{pi}(s)$ - **state-value function** for state $s$ with policy $\pi$.

$q_{pi}(s)$ - **action-value function** for state $s$ with policy $\pi$.

# Markov Decision Process

The lesson starts with **Markov Process (Chain)**, defined by probability transition matrix from state $s$ to state $s'$:

$$\mathcal{P}_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s] $$


# Markov Reward Process

Then to **Markove Reward Process**, adding a reward function $\mathcal{R}_s$ and discount factor $\gamma$:

$$
\begin{aligned}
\mathcal{R}_s &= \mathbb{E}[R_{t+1}\mid \mathcal{S} = s_t] \\
\gamma &\in [0, 1]
\end{aligned}
$$

Then to the **Bellman equation** for MRP:

$$ 
\begin{aligned}
v(s) &= \mathbb{E}[G_t \mid \mathcal{S}_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid \mathcal{S}_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid \mathcal{S}_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma v(\mathcal{S}_{t+1}) \mid \mathcal{S}_t = s] \\
&= \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'} v(s')
\end{aligned}
$$

In matrix form, the Bellman equation is:

$$ v = \mathcal{R} + \gamma \mathcal{P} v $$

where `v.shape = (N, 1)` for `N` states.

To solve for $v$, we have:

$$
\begin{aligned}
(I - \gamma \mathcal{P}) v &= \mathcal{R} \\
v &= (I - \gamma \mathcal{P})^{-1} \mathcal{R}
\end{aligned}
$$

Because this invovles inverting matrices, the complexity is $O(n^3)$ for $n$ states. For large MRPs, need other methods such as:

* Dynamic programming
* Monte Carlo evaluation
* Temporal-Difference learning

# Markov Decision Process

## Adding Action

Then finally to the **Markov Decision Process**, aka **MDP** by adding actions $\mathcal{A}_t$, changing the transition probability matrix to:

$$\mathcal{P}^a_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s, \mathcal{A}_t = a] $$

And changing reward function to:

$$\mathcal{R}^a_s = \mathbb{E}[R_{t+1} \mid \mathcal{S} = s_t, \mathcal{A}_t = a] $$

## Adding Policy

A policy $\pi$ is a distribution over actions given states:

$$ \pi(a \mid s) = \mathbb{P}\big[ A_t = a \mid S_t = s \big] $$

* A policy fully defines the behaviour of an agent.
* MDP policies depend on the current state (not the history)
    - i.e. Policies are stationary (time-independent), $A_t \sim \pi(\cdot \mid S_t), \forall t > 0$

Given an MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \gamma \rangle$ and a policy $\pi$:

* Markov process is defined as $\langle \mathcal{S}, \mathcal{P}^{\pi} \rangle$
* MRP is defined as $\langle \mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma \rangle$

Where:

$$
\begin{aligned}
\mathcal{P}^{\pi}_{s,s'} &= \sum_{a \in \mathcal{A}} \pi(a \mid s)\mathcal{P}^{a}_{ss'}\\
\mathcal{R}^{\pi}_s &= \sum_{a in \mathcal{A}} \pi(a \mid s)\mathcal{R}^a_s
\end{aligned}
$$

## Value Function

**State-value function** of an MDP is the expected return starting from state $s$, and then following policy $\pi$:

$$ v_{\pi}(s) = \mathbb{E} [G_t \mid S_t = s] $$

**Action-value function** $q_{\pi}(s, a)$, is the epxected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$ q_{\pi}(s, a) = \mathbb{E}[G_t \mid S_t=s, A_t=a] $$

**Bellman Expectation Equation**

The state-value function can be decomposed into **immediate reward** plus **discounted value** of successor state:

$$ v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] $$

The action-value function can be decomposed as:

$$ q_{\pi}(s, a) = \mathbb{E}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a] $$

Therefore, the **Bellman Expectation Equation** for $V^{\pi}$is defined as: 

$$ v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) q_{\pi}(s, a) $$

and for $Q^{\pi}$:

$$ q_{\pi}(s, a) = \mathcal{R}^a_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} v_{\pi}(s') $$

Substitute $q_{\pi}(s, a)$ into $v_{\pi}(s)$, we have:

$$ v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \bigg( \mathcal{R}^a_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} v_{\pi}(s') \bigg)$$

And substitute $v_{\pi}(s)$ into $q_{\pi}(s, a)$:

$$ q_{\pi}(s, a) = \mathcal{R}^a_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} \sum_{a \in \mathcal{A}} \pi(a' \mid s') q_{\pi}(s', a') $$

In matrix form:

$$ v_{\pi} = \mathcal{R}^{\pi} + \gamma \mathcal{P}^{\pi} v_{\pi} $$

with direction solution:

$$ v_{\pi} = (I - \gamma \mathcal{P}^{\pi})^{-1} \mathcal{R}^{\pi} $$


## Optimal Value Function

Two optimal value function, **state-value** $v_*(s)$ and **action-value** $q_*(s)$ functions, they are the maximium value functions overall policies.

Once we know the optimal value functions, the MDP is **solved**.

Partial ordering over policy:

$$ \pi > \pi' \text{ if } v_{\pi}(s) \geq v_{\pi'}(s), \forall s $$

**For any MDP, there exists an optimal policy $\pi_*$, and all optimal policies achieve the optmal state-value and action-value function.**

$$
\begin{aligned}
v_{\pi_*}(s) &= v_*(s) \\
q_{\pi_*}(s) &= q_*(s) 
\end{aligned}
$$

**Bellman Optimality Equations**:

$$
\begin{aligned}
v_{*}(s) &= \underset{a}{\max} q_*(s, a) \\
q_{*}(s, a) &= \mathcal{R}^a_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} v_*(s') \\
v_{*}(s) &= \underset{a}{\max} \bigg( \mathcal{R}^a_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} v_*(s') \bigg)
\end{aligned}
$$

The Bellman Optimality Equation:
* is **non-linear**
* has no closed form solution in general
* Can be solved iteratively
    - Value/Policy iteration / Q-learning / Sarsa


# Dynamic Programming

How to read the charts on slide 18? x and y-axes show no. of cars at each location. Each point on the chart is a state. The contour plot shows the policy, e.g. 1 means move 1 car from location A to location B. 

**Synchronous DP algos**:

| Problem | Bellman Equation | Algorithm |
|--------|--------|-------|
| Prediction | Bellman Expectation Equation | Iterative, Policy Evaluation | 
| Control | Bellman Expectation Equation + Greedy Policy improvement | Policy Iteration | 
| Control | Bellman Expectation Equation | Value Iteration | 

* Complexity $O(m n^2)$ if based on state-value function, for $m$ actions and $n$ states
* Complexity $O(m^2 n^2)$ if based on action-value function

# Model-Free Prediction

Goal is to estimate the value function of an **unknown** MDP.

## Monte Carlo 

Instead of iterating through all states, sample from stats under policy ${\pi}$. Instead of computing the expectation of value function, use **empirical** mean return instead of **expected** return. 

Relies on law of large numbers for empirical mean to converge to true mean. Variance reduces in the order of $1/N$ where $N$ is the number of samples.

**Caveat**: MC can only be applied to **episodic** MDPs. All epsidoes **must** terminate.

Keeps a running mean of value functions at each state visited. See slides 8 and 9 for algorithms, during one of many episodes:

* **First-visit** MC policy eval, only eval the first time step $t$ visiting a state:
    - increse counter $N(s) \leftarrow N(s) + 1$
    - Increase total return $S(s) \leftarrow S(s) + G_t$, where $G_t$ is the **return**, see definiton above.
    - Value is estimated by mean return $V(s) = S(s) / N(s)$
* **Every-visit** MC policy eval, counter increased every time a state is visted, return is estimated at every visit. Same procedures as above.

Which is better? Depends on the setting, will visit during TD section.

**Running Mean** for online training

$$
\begin{aligned}
\mu_k &= \frac{1}{k} \sum^k_{j=1} x_j \\
&= \frac{1}{k} \bigg(x_k + \sum^{k-1}_{j=1} x_j \bigg) \\
&= \frac{1}{k} \big(x_k + (k - 1) \mu_{k-1}\big) \\
&= \mu_{k-1} + \frac{1}{k} (x_k - \mu_{k-1})
\end{aligned}
$$

In code:

```{python}
def running_mean(running_mu, k, data):
    mu_new = running_mu + 1 / (k + 1) * (data - running_mu)
    return mu_new, k+1
```

With this running mean algo, the total return calculation becomes:

$$ V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}\big(G_t - V(S_t)\big) $$

In **non-stationary** problems, it can be useful to track a running mean, i.e. forget old episodes:

$$ V(S_t) \leftarrow V(S_t) + \alpha \big(G_t - V(S_t)\big) $$

## Temporal Differencing

At every time step $t$, update the value $V(S_t)$ of the state towards **estimated** return, $R_{t+1} + \gamma V(S_{t+1})$.

* TD learns directly from episodes and non-episodes of experience.
* TD is also model-free
* TD learns from **incomplete** epsidoes, by **bootstrapping**

Goal is the same, to learn $v_\pi$ online from experience under policy $\pi$. 

To compare MC with TD(0):

| Method | Value Update | When |
|--------|--------------|------|
| MC | $V(S_t) \leftarrow V(S_t) + \alpha \big(G_t - V(S_t)\big)$ | Only after final outcome |
| TD(0) | $V(S_t) \leftarrow V(S_t) + \alpha \big(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\big)$ | Every step |

Definitions:

**TD Target / Estimated Return**: $R_{t+1} + \gamma V(S_{t+1})$

**TD Error**: $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

**Essentially, MC uses return $G_t$, TD(0) uses TD Target.**

### Batched MC & TD

Slide 25 problem. $V(B) = 6 / 8 = 0.75$. But $V(A)$ has different solutions under MC ($V(A) = 0$) and TD(0) ($V(A) = .75$).

MC solution is easy to see as we only saw one sample of A and the reward was 0.

TD implicitly solves an MDP, becaues the estimate for $V(B) = 0.75$, and there is 100% probability to go from $A$ to $B$, $V(A) = 0 + V(B) = 0.75$. 

* MC converges to solution with **minimum MSE**.
* TD(0) converges to solution of **max likelihood Markov model**. 
    - TD exploits Markov property, usually **more efficient** in Markov environments.

Good question on if what David meant by non-Markov environment, does MDP
still apply here?

Answer: To clarify, the underlying process may be MDP, but **what the agent observes**, which is what matters to us, may not be. In these environments, MC may be better. 

To compare the sampling and update process, slide 29 shows the MC process, a sample is the entire branch highlighted in red. Once we reach to end state, we update **all** previous states. 

Slide 30 shows the TD(0) sampling path, which is a 1-step look ahead, and immediate update the value for the state we are in.

### $n$-Step Return 

TD($\infty$) is the same as MC.

TD($n$) - **which $n$ is best?** Study showed different choices of $n$ for Online / Offline modes. 

**Average n-Step Returns** - instead of using only one $n$. Can we consider all $n$? This results in TD($\lambda$), essential weights returns at different steps by $(1 - \lambda)\lambda^{n-1}, \lambda \in [0, 1]$

Therefore the return becomes:

$$ G^\lambda_t = (1 - \lambda)\sum^\infty_{n=1} \lambda^{n-1} G^{(n)}_t $$

Value update becomes:

$$ V(S_t) \leftarrow V(S_t) + \alpha \big(G^\lambda_t - V(S_t)\big) $$

Question: Why geometric weights? Answer: computationally efficient, no need to remember previous $\lambda$ state, same cost as TD(0).

**Backward View TD($\lambda$)** can be implemented more efficiently, updates online, every step, from incomplete sequences.

### Eligibility Traces

Credit assignment problem: how much error does each state cause?

Summary table on slide 55.



### Advantages & Disadvantages of MC vs TD

* TD can learn **before** knowing the final outcome
* TD can learn **without** the final outcome
    - TD can learn from **incomplete** sequences, MD needs **complete** sequences.
    - TD works in continuing (non-terminating) environments, not MC.
* MC has **high variance, zero bias**
    - Not sensitive to initial value
    - Good convergence properties (even with function approximation)
    - Usually more efficient in non-Markov environments
* TD has **low variance, some bias**
    - More sensitive to initial value
    - Usually more efficient than MC
    - TD(0) converges to $v_\pi (s)$ (but not always with function approximation), there are some cases where the bias causes problems.
    - Exploits Markov property, usually more efficient in Markov environments.

Good question asked about the upward sloping TD RMS error curve. Answer: this is noise due to $\alpha$ being too large, therefore it oscillates and does not settle down. Normally $\alpha$ needs to be scaled down as the no. of epsidoes increases.


| Method | DP | MC | TD |
|--------|----|----|----|
| Non-Markov Environment Ovserved | NA | Better | Worse |
| Need Complete Sequence | NA (my sense yes) | Yes | No|
| Bootstrapping | Yes | No | Yes |
| Bias | No (my sense) | No | Yes |
| Variance | NA | High | Low |
| Sensitive to Initial Values | No (my sense) | No | Yes |
| Converges to | NA | Min MSE | Max Likelihood Markov |


# Model-Free Control

**On-policy** learning: learn about policy $\pi$ from experience sampled from $\pi$. 

**Off-policy** learning: learn about policy $\pi$ from experience sampled from $\mu$.

With **greedy** policy improvments over $V(S)$ requires model of MDP, which means it **does not** work if we want to do **model-free** learning. (Need probability transition matrix to update state-value function.)

Thus, we have to try **greedy** policy imporovement over **action-value** function $Q(s, a)$, which is model free.

## $\epsilon$-Greedy Exploration

With probability $\epsilon$ to choose a random action, otherwise with probability $1 - \epsilon$ to choose the greedy action.

Theorem to prove that $\epsilon$-greedy policy will find the optimal policy on slide 14.

## Monte-Carlo Control

Every **episode**:

* Policy evaluation, MC policy evaluation, $Q \approx q_{\pi}$
* Policy improvement, $\epsilon$-greedy update

**GLIE**: Greedy in the Limit with Infinite Exploration, slide page 17-18. Theorem: GLIE MC control converges to the optimal action-value function, $q_*(s, a)$.

A subtle issue here is that, when we estimate / update $Q(S_t, A_t)$, we are not really computing the mean for a random I.D. variable. Because policy is being improved upon, it changes over time. 

GILE Algo: 

Sample $k$th epsidoe using $\pi$. For each state $S_t$ and action $A_t$ in the **epsidoe**:

$$
\begin{aligned}
N(S_t, A_t) &\leftarrow N(S_t, A_t) + 1 \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}\big( G_t - Q(S_t, A_t)\big)
\end{aligned}
$$

Given $N(S_1, A_1) = 1$, this algo is not sensitive to initialisation of $Q$, i.e. at the start $G_t$ is used as the action-value.

Improve policy based on new action-value function:

$$
\begin{aligned}
\epsilon &\leftarrow 1 / k \\
\pi &\leftarrow \epsilon-\text{greedy}(Q)
\end{aligned}
$$


## MC vs TD Control

### SARSA

**SARSA** algo: for state $S$ and action $A$, sample the environment to compute the reward $R$, arriving at a new state $S'$, and follow current policy to choose action $A'$.

Every **time-step**:

* Policy evaluation, SARSA, $Q \approx q_{\pi}$
* Policy improvement, $\epsilon$-greedy policy improvement

Update formula:

$$ Q(S, A) \leftarrow Q(S, A) + \alpha \big(R + \lambda Q(S', A') - Q(S, A)) $$

Theorem states that for Sarsa to converge we need:

* GLIE sequence of policy $\pi_t(a \mid s)$
* Robbin-Monro sequence of step-sizes $\alpha_t$

David Silver: In practice, we often ignore Robbin-Monro, and sometimes even GLIE, Sarsa converges anyway. 

n-Step Sarsa is quite similar to TD(n) to update n-step Q-return:

$$
\begin{aligned}
q^{(n)}_t &= R_{t+1} + \gamma R_{t+2} + \cdots + \lambda^{n-1}R_{t+n} + \lambda^n Q(S_{t+n}) \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha \big(q^{(n)}_t - Q(S_t, A_t)\big)
\end{aligned}
$$

### Sarsa($\lambda$)

Similarly, to find the best $n$, use **Sarsa($\lambda$)**, using weight $(1-\lambda)\lambda^n$, **forward-view** below:

$$
\begin{aligned}
q^{\lambda}_t &= (1 - \lambda)\sum^{\infty}_{n=1} \lambda^{n-1} q^{(n)}_t \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \alpha \big(q^{\lambda}_t - Q(S_t, A_t)\big)
\end{aligned}
$$

But again, foward-view is **not** an **online** algorithm. So we need a backward-view version. To do so, we need to build **eligibilty traces** again.

Sarsa($\lambda$) has **one** eligibility trace for **each state-action pair**.

$$
\begin{aligned}
E_0(s, a) &= 0 \\
E_t(s, a) &= \gamma \lambda E_{t-1}(s, a) + \mathcal{1}(S_t = s, A_t = a)
\end{aligned}
$$

**Identificaton function** $\mathcal{1}(S_t = s, A_t = a)$ means when in state $s$ taking action $a$, the function returns 1, i.e. increase the eligibility trace by 1.

Then, the update $Q(s, a)$ for every states $s$ and actions $a$:

$$
\begin{aligned}
\delta_t &= R_{t+1} + \lambda Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\
Q(s, a) &\leftarrow Q(s, a) + \alpha \delta_t E_t(s, a) \text{  # update all states}
\end{aligned}
$$

Gridworld example on page 33:

**Left** picture is an example of Sarsa, i.e. only the last step gets the update when we reach an award. Previous steps get 0.

With **Sarsa**, when a path is sampled and reaches the reward, only the last action-state before the reward would be updated. Previous steps would only see 0 reward. In the next iteration, when the reward is reached, the reward then starts to propagate back one step further. Therefore, update propagates back very slowly, you'd need **lots** of iterations to update all the states. 

Right picture is Sarsa($\lambda$), the size of the arrow indicates the value of eligibility trace. 

With **Sarsa($\lambda$)**, when the reward is reached, **all** previous action-state pairs would be updated, according to the eligibility traces. Defeats the **tyrany of time steps**.

Question: why wait until seeing the reward to update Q's?

A: great question. Actually the updates happen every step, they just contain zero information. Only when you reach the reward, would you **gain information**. 

### Off-Policy Learning

Evaluating **target policy** $\pi(a \mid s)$ to compute $v_\pi(s)$ or $q_\pi(s, a)$, while folloing **behaviour policy** $\mu(a \mid s)$.

#### Importance Sampling

Estimating the expectation of a different distribution. 

$$
\begin{aligned}
E_{X \sim P} [f(X)] &= \sum P(x)f(X) \\
&= \sum Q(X)\frac{P(X)}{Q(X)}f(X) \\
&= E_{X \sim Q} \bigg[\frac{P(X)}{Q(X)}f(X) \bigg]
\end{aligned}
$$

MC off-policy learning with importance sampling is a really **bad** idea. Extremely high variance.

TD off-policy: use TD targets generated from $\mu$ to evaluate $\pi$. 

* Weight TD target $R + \gamma V(S')$ by importance sampling
* Only need a single importance sampling correction:

$$ V(S_t) \leftarrow V(S_t) + \alpha\bigg( \frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)} (R_{t+1} + \gamma V(S_{t+1})) - V(S_t)\bigg) $$

TD learning still has high variance, can still blow up, but much better than MC.

#### Q-Learning

Better than TD with importance sampling. 

* Consider off-policy learning of action-values $Q(s, a)$.
* Next action is chosen using **behaviour policy** $A_{t+1} \sim \mu(\cdot \mid S_t)$
* But consider **alternative** successor action $A' \sim \pi(\cdot \mid S_t)$
* Then update $Q(S_t, A_t)$ towards value of **alternative action**:

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big(R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t) \big) $$

The special case of this is **Q-Learning** (**SARSAMAX**): **target policy** $\pi$ is **greedy** w.r.t. $Q(s, a)$:

$$ \pi(S_{t+1}) = \underset{a'}{\operatorname{argmax}} Q(S_{t+1}, a') $$

Essentially, go with S, A, R, $S'$, then choose the max of $A'$:

$$ Q(S, A) \leftarrow Q(S, A) + \alpha \big(R + \gamma \underset{a'}{\operatorname{max}}Q(S', a') - Q(S, A) \big) $$

**Theorem**: Q-learning control converges to the optimal action-value function, $Q(s, a) \rightarrow q_*(s, a)$

**Great slide 46 compares DP methods with TD/Q-learning.**

| Full Backup (DP) | Sample Backup (TD) |
|------------------|--------------------|
| Iterative Policy Evaluation <br> $V(s) \leftarrow E[R +\gamma V(S') \mid s]$ | TD Learning <br> $V(S) \overset{a}{\leftarrow} R + \gamma V(S')$ |
| Q-Policy Iteration <br> $Q(s, a) \leftarrow E[R + \gamma Q(S' A') \mid s, a]$ | Sarsa <br> $Q(A, S) \overset{a}{\leftarrow} R + \gamma Q(S', A')$ |
| Q-Value Iteration <br> $Q(s, a) \leftarrow E\bigg[R + \gamma \underset{a' \in \mathcal{A}}{\operatorname{max}} Q(S', a') \mid s, a\bigg]$ | Q-Learning <br> $Q(S, A) \overset{a}{\leftarrow} R + \gamma \underset{a' \in \mathcal{A}}{\operatorname{max}} Q(S', a')$ |

Where $x \overset{a}{\leftarrow} y \equiv x \leftarrow x + \alpha (y - x)$, $\equiv$ indicates **equivalence relation / identical**. 


# Value Function Approximation

In RL we require training methods that is suitable for **non-stationary, non-iid** data.

**Notation**: functions with $\hat{q}$ or $\hat{v}$ are function approximations of the true values of $q$ and $v$. 

## Incremental Methods

In RL, there is **no supervisor, only rewards**. Hence, in practice, we substitue a **target** for $v_\pi(s)$. 

* MC - target is the return $G_t$
* TD(0) - target is the TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$
* TD($\lambda$) - the target is the $lambda$-return $G^{\lambda}_t$

### MC

MC evaluation converges to a **local optimum**, even when using non-linear function approximation.

### TD(0)

Linear TD(0) converges (close) to **global optimum**.

### TD($\lambda$)

Slide 25, the summary of these is that between TD(0) ($\lambda=0$) and MC ($\lambda=1$), there is usually a sweet spot (black solid lines).

Biased estimate, but has been proven that it can still converge.

The dimension of **eligibility traces** is the same as the features that represent the state.

Question at around 46-mins: On slide 17, why the does the TD(0) gradient only contain $\hat{v}(S, w)$, not $\hat{v}(S', w)$? Note: $\hat{v}$ means it's a function approximation of the true $v$.

A: Family of techniques known as **Residual Gradient Methods**. Do not want to reverse time and update the step $t+1$ towards $t$ where you haven't seen the result. Don't really undertand it.... Feels like a hack as the derivative of the loss function w.r.t. weights would look at both terms.

* TD does not follow the gradient of **any** objective function. 
* This is why TD can diverge when off-policy or using non-linear function approximiation.
* **Gradient TD** follows true gradient of projected Bellman error

### Action-Value Function Approximation / Incremental Control

Subtitute a target for $q_\pi(S, A)$, like prediction / state-value function approximation.

Goal is to minimize the squared error between the function approximation $\hat{q}(S, A, w)$ and the true values $q(S, A, w)$.

<a id='pred_convergence'></a>
### Convergence of Prediction Algo

| On / Off-policy | Algo | Table Lookup | Linear | Non-Linear |
|---|:---:|---|---|-----|
| On-Policy | MC | Yes | Yes | Yes |
| On-Policy | LSMC | Yes | Yes |  |
| On-Policy | TD(0) | Yes | Yes | No | 
| On-Policy | TD($\lambda$) | Yes | Yes | No |
| On-Policy | LSTD | Yes | Yes |  |
| On-Policy | Gradien TD | Yes | Yes | Yes |
| Off-Policy | MC | Yes | Yes | Yes |
| Off-Policy | LSMC | Yes | Yes |  |
| Off-Policy | TD(0) | Yes | No | No | 
| Off-Policy | TD($\lambda$) | Yes | No | No |
| Off-Policy | LSTD | Yes | Yes |  |
| Off-Policy | Gradien TD | Yes | Yes | Yes |

| On / Off-policy | Algo | Table Lookup | Linear | Non-Linear |
|---|:---:|---|---|-----|
| On-Policy | MC | Yes | Yes | Yes |
| On-Policy | TD(0) | Yes | Yes | No | 
| On-Policy | TD($\lambda$) | Yes | Yes | No |
| On-Policy | Gradien TD | Yes | Yes | Yes |
| Off-Policy | MC | Yes | Yes | Yes |
| Off-Policy | TD(0) | Yes | No | No | 
| Off-Policy | TD($\lambda$) | Yes | No | No |
| Off-Policy | Gradien TD | Yes | Yes | Yes |

<a id='control_convergence'></a>
### Convergence of Control Algos

| Algo | Table Lookup | Linear | Non-Linear |
|:---:|---|---|-----|
| MC Control | Yes | (Yes) | No |
| Sarsa | Yes | (Yes) | No |
| Q-Learning | Yes | No | No |
| Gradient Q-Learning | Yes | Yes | No |
| LSPI | Yes | (Yes) | No |

(Yes) - chatters around near optimal value function.


# Batch Methods

## Experience Reply

Finds the least square (min MSE) solution.

* Sample (state, value) from experience
* Apply SGD update with this sample as data input

Experience replay **decorrelates the original non-iid data** (as data is sampled from a memory / dataset), therefore, stablise non-linear methods in training.

## Deep Q-Networks (DQN)

Uses **experience replay** and **fixed Q-targets**. Algo:

* Take action $a_t$ according to $\epsilon$-greedy policy
* Store transition $(s_t, a_t, r_{t+1}, s_{t+1})$ in replay memory $\mathcal{D}$
* Sample mini-batches of size 64 of transitions $(s, a, r, s')$ from $\mathcal{D}$
* Compute Q-learning targets w.r.t. **old, fixed parameters $w^-$**, i.e. fixed Q-targets.
* Optimize MSE between Q-network (action-value function) and Q-learning targets (from the old network with $w^-$)

$$\mathcal{L}_i(w_i) = \mathbb{E}_{s,a,r,s' \sim D}\bigg[ \big(r + \gamma \underset{a'}{\max} Q(s', a'; w^-_i) - Q(s, a; w_i)\big)^2 \bigg] $$

* Using variant of SGD. 

DQN is **stable** with NN / Non-linear methods, handles the issues mentioned before with non-linear function approximators. Due to:

* Experience reply (handling non-iid data)
* Keep two Q-Networks, bootstrap towards an older network a few thousands steps ago. I.e. use old network with params $w^-$ (frozen) to compute Q-learning targets, but update gradients in the new network with params $w$. Every few thousands steps after, swap replace the old network with the latest / current network.

Question asked at 1h22min on this two-network approach more. If we didn't keep the old network fixed and just had one network, while we are doing SGD, the parameters of the network get updated with each data sample / batch, which means that the Q-targets keep moving as well. This causes problem for the training process and can blow up.

## Linear Least Square Prediction

* For N features, direction solution with inverting a maxtrix is $O(N^3)$
* Incremental solution time is $O(N^2)$ using Shermann-Morrison

Flavours available:

* LSMC - with MC returns
* LSTD - with TD targets
* LSTD($\lambda$) - use TD($\lambda$) $\lambda$-return

**Convergence** of LS prediction algos [here](#pred_convergence) (for LS value prediction).

To do control, use LSPI (Least Square Policy Iteration). **Convergence** of LS control algos [here](#control_convergence) (for LS control).


# Policy Gradient

There are cases where representing policy can be more compact then representing values.

**Advantages**:

* Better convergence properties 
* Effective in high dimensional or continuous action spaces (With value methods, we need to compute **max** of some values, which can be expensive)
* Can learn stochastic policies (when **state aliasing** occurrs, i.e. partially observed environment, or feature space cannot fully represent states, stochastic policies are better than deterministic policies)

**Disadvantages**:

* Typically converge to a **local** rather than **global** optimum
* Evaluating a policy is typically inefficient and **high** variance (naive methods)

## Policy Objective Function

In episodic environments, we cna use start value

$$ J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[v_1] $$

In continuing environments we can use **average value**

$$ J_{av}v(\theta) = \sum_s d^{\pi_\theta}(s)V^{\pi_\theta}(s) $$

Or the **average reward per time-step**:

$$ J_{av}v(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s, a)\mathcal{R}^a_s $$

$d^{\pi_\theta}(s)$ is a stationary distribution of Markov chain for $\pi_\theta$, i.e. the probability of being in state $s$ under policy $\pi_\theta$.

Goal: find $\theta$ that **maximises** $J(\theta)$


## Score Function

**Likelihood ratio**:

$$ 
\begin{aligned}
\triangledown_\theta \pi_\theta(s, a) &= \pi_\theta(s, a) \frac{\triangledown_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} \\
&= \pi_\theta(s, a) \triangledown_\theta \log \pi_\theta(s, a)
\end{aligned}
$$

This is because $\triangledown_x(log x) = \frac{1}{x}$, let $x = \pi_\theta(s, a)$ and using the chain rule, we have:

$$ \frac{d}{d\theta} \log \pi_\theta(s, a) = \frac{1}{\pi_\theta(s, a)} \frac{d}{d\theta} \pi_\theta(s, a) $$

**Score function** is $\triangledown_\theta \pi_\theta(s, a)$

## Softmax Policy

$\phi(s, a)$ is a feature. Score function is feature minus average of all features:

$$ \triangledown_\theta \log \pi_\theta(s, a) = \phi(s, a) - \mathbb{E}_{\pi_\theta} [\phi(s, \cdot)] $$

## Gaussian Policy

Natural for continuous action space. 

Parameterised by mean and variance. 

Mean is linear combination of state features $\mu(s) = \phi(s)^T \theta$

Variance can be fixed $\sigma^2$ or also parameterised. 

Score function is:

$$ \triangledown_\theta \log \pi_\theta(s, a) = \frac{(a - \mu(s))\phi(s)}{\sigma^2} $$

## Policy Gradient Theorem

For any differentiable policy $\pi_\theta(s, a)$, for any of the policy objective, $J = J_1$, $J_{av}R$, or $\frac{1}{1-\gamma}J_{av}V$, the policy gradient is:

$$\triangledown_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \big[ \triangledown_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a) \big] $$

Where $Q^{\pi_\theta}(s, a)$ is the action-value function. This is still **model-free**, i.e. we can sample $(s, a)$.

Intuitively, this is: expectation of how to improve our policy times the reward we see. 


## Actor-Critic Methods


