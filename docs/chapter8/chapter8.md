# Deep Reinforcement Learning

<!-- TODO: is word store correct ? always try to make the writing clearer, now just go with the flow -->
<!-- Explain how state scaling fails-->

In the previous chapters we were using tabular methods to store our expected rewards, either if it was for individual states or for every state-action pairs. Now imagine 

## Deep Q Learning
<!-- TODO: NOT DONE and written really badly -->
A quick recap on Q-learning, so in Q-learning we try to learn $Q*$, the optimal quality function, and if we visit every state-action pairs infinite many times, eventually we will converge to $Q*$.

$$Q(s, a)_{k+1} = Q_k(s, a) + \alpha[r_{t+1} + \gamma \max_{a'} Q(s', a') - Q_k(s, a)]$$

<!-- TODO: NOT DONE and maybe talk about a known game, not just numbers -->
As the state space grows, tabular Q-learning becomes computationally impossible. Consider a state space of $10^6$ states with 10 actions, the table alone requires $10^7$ entries, and convergence requires every one of those entries to be visited infinite many times. In practice, most states are never visited at all, making no convergence guarantees.

<!-- TODO: NOT DONE and be more precise about are we changing the values of s and a, not s only  -->
<!-- TODO: put a link to the last chapter maybe ?-->
The problem with tabular Q-learning is that we are learning each state individually, and modifying the value of state $s$ would never change the value any other state $s'$ in any way. When using function approximators changing a parameter in the function changes the function landscape, leading to the modification of multiple state values. In the last chapter we approximated our $Q$ function using linear approximators, but still linear approximators encode the states using feature vectors that are designed by us rather than learned from experience.

The usage of deep neural nets here is our best option, and that is exactly how in deep Q-learning we approximate our Q function, deep neural learns the feature representation of a state by its own,

### Playing Atari with Deep RL

<!-- TODO: proof read, this is badly written -->
Consider an example where we are trying to train an agent to play the atari game breakout, our state here is the current image of the game, so using tabular methods is just impossible, instead we will be using a deep neural network, so now our $Q$ function will be parameterized by some weights $\theta$. Our goal becomes making our $Q$ function eventually learn the optimal $Q$ function.

$$Q(s, a; \theta) \approx Q^*(s, a)$$

<!-- Write about how does normal ML differ from Deep RL in the idea of the target  -->
The main difference from supervised ML is that here our target is not fixed, we don't have a true value that we are trying to converge, instead we have a moving target that is changing every update of the weight, and this happens due to us using temporal difference estimates

## Double DQN

## Dueling Networks

## Prioritized replay
