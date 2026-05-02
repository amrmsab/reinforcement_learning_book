# Chapter 3: Dynamic Programming

---

## 3.1 Dynamic Programming

Dynamic Programming is an algorithmic technique used for solving complex problems by breaking them into smaller overlapping simpler subproblems.

The term dynamic programming in reinforcement learning, however, refers to algorithms that can compute optimal policies given a perfect model of the environment as an Markov Decision Process. These DP algorithms only work because of the assumption that a perfect model is already present. Another assumption is that we are working Another important factor Dp contributes to is that it provides an essential foundation to newer, more complex methods of understanding reinforcement learning.

In addition, those methods offer to provide the same effect as Dynamic Programming with less computation time and the lack of the perfect model assumption.

The importance of Dynamic Programming in Reinforcement Learning is how DP uses the value functions to search for the optimal policy. DP applies the Bellman Equation repeatedly to compute both the value function and the optimal policy.

In Reinforcement Learning, DP can help us Evaluate a policy, improve it and directly compute optimal values, using Policy Evaluation, Policy Iteration, and Value Iteration respectively.

---

## 3.2 Policy Evaluation

Policy Evaluation asks the question of how good the current policy is. A policy, *π*(a|s), is the probability of taking an action *(a)* in a specific state *(s)*. Evaluating a Policy is the process of solving the equation

$$\sum_a \pi(a|s) \sum_{s',r} p(s', r|s, a)\left[r + \gamma v_{\pi}(s')\right]$$

and it is done trivially, while being computationally heavy depending on the number of states and actions, using DP only when the full model of the world is known, which is the assumption made when using DP. And it can be shown that the equation can be solved iteratively and will eventually converge on $v_{\pi}$. Iterative Policy Evaluation will continue to update the states with new values from the value function.

![Figure 3.2 — Policy Evaluation](Policy%20Evaluation.png)

---

## 3.3 Policy Improvement

The main reason we compute the value function for a policy is to find better, more efficient policies. The process of making a policy that improves on the original policy is called **Policy Improvement**. By performing *Policy Improvement* we can essentially **iterate** through many policies and eventually select the best and most optimal policy.

![Figure 3.3 — Policy Improvement](Policy%20Improvement.png)

The Figure above shows the use of an arbitrary policy in k=0, and eventually changes policy with every iteration and eventually settles on the optimal policy after multiple iterations when noticing that the ratios of values is almost identical.

---

## 3.4 Policy Iteration

Policy Iteration combines the knowledge of both Policy Evaluation and Policy Improvement into one recursive process. The first step is to initialize the value function with an arbitrary policy for all states. The second step is to evaluate the arbitrary policy using the value function. The third step is to choose a new policy. Finally, repeat until the optimal policy is found.

![Figure 3.4 — Policy Iteration Steps](Policy%20Iteration%20steps.png)

Policy Iteration terminates in a finite amount of steps due to the MDP we work with being finite itself. Meaning we have a finite number of policies to work with. However, if sometimes two or more policies provide equally good solutions, policy iteration may not terminate as it keeps switching between these equally good solutions, in which the algorithm must change to account for such scenarios.

---

## 3.5 Value Iteration

An important drawback of Policy Iteration, is that each iteration involves policy evaluation. Multiple policy evaluations through the set of actions available is computationally heavy and those computations may be truncated by using a greedy algorithm that chooses the optimal policy at each state *s*. This kind of algorithm is referred to as *value iteration*.

Value Iteration assumes the agent always acts as optimally as possible. It effectively improves on Policy Iteration by doing one sweep of value evaluation and one sweep of policy improvement in one iteration. A sweep is basically a comparison between all value functions and selecting the best policy during the comparison. Value Iteration terminates once the value function changes by only a small amount in a sweep.

![Figure 3.5 — Value Iteration](Value%20Iteration.png)

---

## 3.6 Asynchronous Dynamic Programming

The use case for the Dynamic Programming methods explained so far is only setback by the size of the entire state set of the MDP. In a very large state set, regular Dynamic Programming may not be efficient computationally. *Asynchronous* DP algorithm helps mitigate this problem. They don't necessarily improve computation, but it creates an algorithm that doesn't get locked in a very long sweep before it can improve the policy.

---

## 3.7 Advantages & Drawbacks of Dynamic Programming

While it's not very practical for very large problems, its efficiency in solving those problems is obvious. For a finite MDP, the worst case for DP to find the optimal policy is polynomial time. DP is exponentially faster than any direct search methods of finding the optimal policy. On problems with large state sets, Asynchronous DP is preferred. Asynchronous methods and other variations in these cases may find good or optimal solutions faster than its synchronous counterpart.

Dynamic Programming has a very glaring limitation compared to these advantages. It requires a **complete and accurate model** of the environment. Usually, RL agents don't have a complete understanding of the model of the environment.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Available at: <http://incompleteideas.net/book/RLbook2020.pdf>
2. van Otterlo, M., & Wiering, M. (2012). Reinforcement Learning and Markov Decision Processes. University of Groningen. Available at: <https://www.ai.rug.nl/~mwiering/Intro_RLBOOK.pdf>