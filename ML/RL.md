# Reinforcement Learning Notes 

Based on Dan Silver's 2016 UCL course, videos [here](#https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

## Notation

$\mathcal{S}$ - a finite set of **states**.

$\mathcal{P}$ - state **transition probability matrix**.

$\mathcal{R}$ - **reward function**.

$\gamma$ - **reward discount factor**.

$G_t$ - **Return**, total discounted reward from time step $t$. $G_t = \sum_{k=0}^{\infty}\gamma^k R_{t + k + 1}$.

$v(s)$ - **value function** for a Markove Reward Process is the **expected return** starting from state $s$.

$\mathcal{A}$ - a finite set of **actions**.

## Markov Decision Process

The lesson starts with **Markov Process (Chain)**, defined by:

$$\mathcal{P}_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s] $$

Then to **Markove Reward Process**, adding a reward function and discount factor:

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

Then finally to the **Markov Decision Process**, aka MDP by adding actions, changing the transition probability matrix to:

$$\mathcal{P}^a_{ss'} = \mathbb{P}[\mathcal{S}_{t+1} = s' \mid \mathcal{S}_t = s, \mathcal{A}_t = a] $$

And changing reward function to:

$$\mathcal{R}^a_s = \mathbb{E}[R_{t+1} \mid \mathcal{S} = s_t, \mathcal{A}_t = a] $$

