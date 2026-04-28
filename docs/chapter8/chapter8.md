# Deep Reinforcement Learning

<!-- TODO: is word store correct ? always try to make the writing clearer, now just go with the flow -->
<!-- Explain how state scaling fails-->

In the previous chapters we were using tabular methods to store our expected rewards, either if it was for individual states or for every state-action pairs. Now imagine 

## Deep Q Learning
<!-- TODO: NOT DONE and written really badly -->
A quick recap on Q-learning, so in Q-learning we try to learn $Q^*$, the optimal quality function, and if we visit every state-action pairs infinite many times, eventually we will converge to $Q^*$.

$$Q(s, a)_{k+1} = Q_k(s, a) + \alpha[r_{t+1} + \gamma \max_{a'} Q(s', a') - Q_k(s, a)]$$

Q-learning is an off-policy learning algorithm since we use $\epsilon$-greedy methods for data generation while the policy that we will be using after training will be:

$$\pi(s) = \arg\max_a Q(s, a)$$

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

<!-- TODO: proof read, this is badly written -->
<!-- TODO: try to link with the linear approximators chapers and use their symbols -->
In the early layers we will be using a convolutional neural network, this network extracts the important features from our current state, so instead of passing the whole complete image of $84 * 84$ pixels as an input for the fully connected layers, the CNN learns to extract the important features needed from that image into a meaningful vector representation that the later layers can make the best use of. Another way to look at the convolutional layers is that they are learning the correct state vector representation for the state analogous to the feature vectors we hand-crafted in linear function approximators. This feature vector is then passed to multiple fully connected layers that have RELU activation function between them and finally in the end we compute the value for every possible action.

<!-- Write about how does normal ML differ from Deep RL in the idea of the target  -->
The main difference from supervised ML is that here our target is not fixed, we don't have a true value that we are trying to converge, instead we have a moving target that is changing every update of the weight, and this happens because we are using the same neural network for action selection and evaluation.

### Replay Buffer

<!-- current level what ? make it more precise please-->
The agent is not trained on sequential steps in the same episode, this causes to the network to overfit to the plays that is optimal to its current level 

<!-- TODO: proof read, this is badly written -->
Another way to look at this is that to image you are training a normal neural network that does classification for cat and dog images, but instead of shuffling your mini-batch, you just select full batch of cat samples, here your network will just optimize for outputting cats whatever the sample, and your training won't converge and you will be forgetting your previous training. The same thing could happen in DQN, the network will just overfit to the current path, forget what it learned before.

## Double DQN

## Dueling Networks

## Prioritized replay
