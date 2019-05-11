# Reinforcement Learning Notes 

Based on Dan Silver's 2016 UCL course, videos [here](#https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

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

### Eligibility Traces

Credit assignment problem: how much error does each state cause?

Summary table on slide 55.
