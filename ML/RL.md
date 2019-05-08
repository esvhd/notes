# Reinforcement Learning Notes 

Based on Dan Silver's 2016 UCL course, videos [here](#https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

## Notation

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

## Markov Decision Process

The lesson starts with **Markov Process (Chain)**, defined by probability transition matrix from state $s$ to state $s'$:

$$\mathcal{P}_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s] $$


## Markov Reward Process

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

## Markov Decision Process

### Adding Action

Then finally to the **Markov Decision Process**, aka **MDP** by adding actions $\mathcal{A}_t$, changing the transition probability matrix to:

$$\mathcal{P}^a_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s, \mathcal{A}_t = a] $$

And changing reward function to:

$$\mathcal{R}^a_s = \mathbb{E}[R_{t+1} \mid \mathcal{S} = s_t, \mathcal{A}_t = a] $$

### Adding Policy

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

### Value Function

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


### Optimal Value Function

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


## Dynamic Programming

How to read the charts on slide 18? x and y-axes show no. of cars at each location. Each point on the chart is a state. The contour plot shows the policy, e.g. 1 means move 1 car from location A to location B. 

**Synchronous DP algos**:

| Problem | Bellman Equation | Algorithm |
|--------|--------|-------|
| Prediction | Bellman Expectation Equation | Iterative, Policy Evaluation | 
| Control | Bellman Expectation Equation + Greedy Policy improvement | Policy Iteration | 
| Control | Bellman Expectation Equation | Value Iteration | 

* Complexity $O(m n^2)$ if based on state-value function, for $m$ actions and $n$ states
* Complexity $O(m^2 n^2)$ if based on action-value function
