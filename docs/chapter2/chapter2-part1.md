# Chapter 6: Temporal-Difference Learning (TD)
## Section 1: Prediction
**Author:** Radwa Ibrahim

## Table of Contents
1. [Introduction: The Best of Both Worlds](#1-introduction-the-best-of-both-worlds)
2. [The Mathematics of TD(0)](#2-the-mathematics-of-td0)
3. [Intuitive Walkthrough: The "Driving Home" Problem](#3-intuitive-walkthrough-the-driving-home-problem)
4. [Why is TD Better?](#4-why-is-td-better)
5. [Deep Dive: The Optimality of TD(0)](#5-deep-dive-the-optimality-of-td0)
6. [Python & Library Suggestions](#6-python--library-suggestions)
7. [Conclusion](#7-conclusion)
8. [References & AI Disclosure](#8-references--ai-disclosure)

---

## 1. Introduction: The Best of Both Worlds

Imagine you are on a long road trip. If you take a wrong turn, do you wait until the end of the trip to realize your mistake, or do you correct your course at the very next exit? 

This is the core difference between previous Reinforcement Learning (RL) methods and **Temporal-Difference Learning**. 

Before we dive in, we must remember the **RL Divide**:
* **Prediction:** Evaluating a policy (How good are my current rules?).
* **Control:** Optimizing a policy (How do I find the *best* rules?).

**This section focuses entirely on Prediction.** We want to estimate the value function $v_\pi$ for a given policy $\pi$. To do this, TD learning seamlessly combines the best aspects of two foundational RL concepts:

1. **Monte Carlo (MC):** Learns directly from raw experience (no environment model required).
2. **Dynamic Programming (DP):** Updates estimates based on other learned estimates—a process known as **bootstrapping**—without waiting for the final outcome.

![TD Venn Diagram](images/Venn-Diagram(DP,MC,TD).png.png) 
*(Caption: TD Learning sits at the exact intersection of Monte Carlo and Dynamic Programming.)*

> 🧠 **Test Your Understanding:**
> Why do we call TD learning a "bootstrapping" method?
> *Answer: Because it learns a guess from a guess! It updates the value of $S_t$ using the estimated value of $S_{t+1}$, rather than waiting for the true final outcome.*
---

## 2. The Mathematics of TD(0)

All incremental learning methods share a similar update rule structure:
> **New Estimate $\leftarrow$ Old Estimate + StepSize $\times$ [Target - Old Estimate]**

Let's look at the side-by-side comparison of how Monte Carlo and the simplest form of TD learning—known as **TD(0)**—handle this update.

![MC vs TD Equations](images/Math-Comparisons.png.png) 
*(Caption: Notice how the only structural difference is what each algorithm uses as its "Target".)*

### Breaking Down the Variables
Before looking at an example, we must understand the parameters that control how our agent learns:
* **Learning Rate ($\alpha$):** This dictates how much we override our old estimate with the new information. 
  * If $\alpha = 0$, the agent learns nothing. 
  * If $\alpha = 1$, the agent completely replaces its old guess with the new target. Typically, we use a small value (like $0.1$) to ensure stable, gradual learning.
* **Discount Factor ($\gamma$):** This dictates how much the agent cares about *future* rewards. 
  * If $\gamma = 0$, the agent is completely short-sighted (only cares about the immediate reward). 
  * If $\gamma = 1$, future rewards are weighted equally to immediate ones (often used in finite episodes, like driving home).

### The Monte Carlo vs. TD Target
* **MC Target:** $G_t$ (The actual total return). MC must wait until the episode ends to calculate this.
* **TD Target:** $R_{t+1} + \gamma V(S_{t+1})$ (Immediate reward + discounted value of the next state). TD only waits *one single time step* before updating.

> 🧠 **Test Your Understanding:**
> The term in the brackets $[Target - Old Estimate]$ is called the **TD Error** ($\delta_t$). 
> *If the TD Error is positive, what does that mean?* 
> It means our experience was better than we expected, so we should increase the value of our previous state!

### A Quick Note on $n$-step TD
You might be wondering why we explicitly call this algorithm **TD(0)**. 

The "0" indicates that we are looking exactly *one* step ahead before bootstrapping. However, TD is actually a flexible, $n$-step algorithm. You could choose to wait 2 steps, 3 steps, or $n$ steps before making your update. 
* If $n = 1$, you have **TD(0)**.
* If $n = \infty$ (waiting until the very end of the episode), TD simply becomes **Monte Carlo**.

For the sake of simplicity, this section focuses entirely on the 1-step TD(0) algorithm, but it is important to remember that TD and MC exist on a continuous spectrum!

---

## 3. Intuitive Walkthrough: The "Driving Home" Problem

Let's ground this math in a real-world example (Sutton & Barto, 2018, p. 145). 

You leave your office on a Friday. You estimate it will take 30 minutes to drive home. As you navigate traffic, rain, and slow trucks, your estimate changes.

**The MDP Setup:**
* **States:** Locations in your commute (Office, Car, Highway, Truck Road, Home Street, Home).
* **Rewards ($R$):** Elapsed time between states.
* **Goal:** Predict total time to get home.

![Driving Home States Setup](images/DriveHome-Example.png.png)
*(Caption: The sequence of states, elapsed times, and our initial predictions for how much time is remaining.)*

### The Monte Carlo Way (Waiting for the End)
With $\alpha = 1$, the MC update formula simplifies to: $V(S_t) \leftarrow G_t$. 
MC must wait until we arrive home at minute 43 to know the actual total returns ($G_t$). Once we arrive, we look back and update every state based on the *actual* elapsed time:

* **$S_0$ (Office):** 
  * Actual Return: $43$ 
  * Update: $V(S_0) \leftarrow 30 + [43 - 30] = \mathbf{43}$
* **$S_1$ (Car/Rain):** 
  * Actual Return: $43 - 5 = 38$ 
  * Update: $V(S_1) \leftarrow 35 + [38 - 35] = \mathbf{38}$
* **$S_2$ (Highway):** 
  * Actual Return: $43 - 20 = 23$ 
  * Update: $V(S_2) \leftarrow 15 + [23 - 15] = \mathbf{23}$
* **$S_3$ (Truck):** 
  * Actual Return: $43 - 30 = 13$ 
  * Update: $V(S_3) \leftarrow 10 + [13 - 10] = \mathbf{13}$
* **$S_4$ (Home Street):** 
  * Actual Return: $43 - 40 = 3$ 
  * Update: $V(S_4) \leftarrow 3 + [3 - 3] = \mathbf{3}$

### The Temporal-Difference Way (Updating on the Go)
To make the math crystal clear, let's set our learning rate **$\alpha = 1$** (we completely trust our new targets) and our discount factor **$\gamma = 1$** (we care equally about all time spent, no discounting).

The TD(0) update formula simplifies to:
$V(S_t) \leftarrow V(S_t) + 1 \cdot [R_{t+1} + 1 \cdot V(S_{t+1}) - V(S_t)]$

Here is how TD updates our estimates *step-by-step* as we drive:

* **Minute 5: Sitting in the car ($S_1$)** 
  * *Action:* We update the Office ($S_0$).
  * $V(S_0) \leftarrow 30 + [5 + 35 - 30] = \mathbf{40}$
* **Minute 20: Exiting Highway ($S_2$)**
  * *Action:* We update the Car ($S_1$).
  * $V(S_1) \leftarrow 35 + [15 + 15 - 35] = \mathbf{30}$
* **Minute 30: Stuck behind Truck ($S_3$)**
  * *Action:* We update the Highway ($S_2$).
  * $V(S_2) \leftarrow 15 + [10 + 10 - 15] = \mathbf{20}$
* **Minute 40: Entering Street ($S_4$)**
  * *Action:* We update the Truck Road ($S_3$).
  * $V(S_3) \leftarrow 10 + [10 + 3 - 10] = \mathbf{13}$
* **Minute 43: Arrive Home ($S_5$)**
  * *Action:* We update the Home Street ($S_4$). State 5 is terminal, so $V(S_5) = 0$.
  * $V(S_4) \leftarrow 3 + [3 + 0 - 3] = \mathbf{3}$

### Comparing the Updates Graphically
The following graphs from Sutton & Barto (2018) beautifully illustrate the difference between the two methods:

![MC vs TD Graphs](images/MC-TD-Gragh.png.png)
*(Caption: Left: MC updates. Right: TD updates. Figures reproduced from Sutton & Barto, 2018.)*

* **In the MC Graph (Left):** The arrows point from our initial predictions all the way to the horizontal "actual outcome" line. Every error is calculated against the final reality of the trip.
* **In the TD Graph (Right):** The arrows point from one state's prediction to the *next state's prediction*. Every error is calculated sequentially based on the temporal difference between current and future guesses.

### ⚠️ The Drawback of 1-Step Bias
Look closely at the update we made when we reached the Truck ($S_3$). To update the Highway ($S_2$), we *only* looked at the immediate reward (10 mins) and the next state's estimate ($V(S_3) = 10$). 

We didn't explicitly look far ahead at the reality of being stuck behind that truck for a long time. TD(0) is heavily biased toward the immediate next state. If we notice that our updates are too heavily swayed by illogical or highly volatile next states, we can handle this by:
1. **Lowering the Learning Rate ($\alpha$):** So we don't completely overwrite our old logic based on one weird transition.
2. **Reducing the Discount Factor ($\gamma$):** If we want the agent to focus strictly on immediate feedback rather than relying too heavily on uncertain future state estimates.

---

## 4. Why is TD Better?

Since TD learning borrows ideas from both Dynamic Programming (DP) and Monte Carlo (MC), it naturally inherits their greatest strengths while avoiding their biggest weaknesses. Here is why TD is widely considered the central method of modern Reinforcement Learning:

### 1. Model-Free Learning (Advantage over DP)
Like MC, TD learning does not require a perfect mathematical model of the environment. You don't need to know the transition probabilities or the exact reward distributions ahead of time. The agent learns simply by interacting with the environment and experiencing raw outcomes.

### 2. Online and Incremental Updates (Advantage over MC)
With Monte Carlo, you must wait until the episode ends to learn anything. TD, however, updates its predictions after *every single step*. This is a massive advantage in two scenarios:
* **Exceptionally Long Episodes:** If an episode takes thousands of steps, delaying learning until the end is computationally slow and inefficient.
* **Continuing Tasks:** Some environments (like a server routing traffic, or a robot balancing indefinitely) have *no terminal state*. MC simply cannot be used here because the episode never ends! TD works perfectly in continuing tasks.

### 3. Efficient Credit Assignment
Because MC waits until the end of the episode to distribute updates, it tends to "smear" the blame (or praise) across every action taken in the sequence. TD fixes this by assigning credit or blame exactly where it belongs.

> 🧠 **Critical Thinking: The Chess Dilemma** 
> Imagine you are playing a game of chess. You play a brilliant game for 39 moves, but on your 40th move, you make a terrible blunder and lose the game. 
> * If you use **Monte Carlo**, the algorithm waits until the game ends (Loss = 0) and penalizes *all 40 moves* you made, failing to recognize that the first 39 were actually great. 
> * With **TD Learning**, the algorithm evaluates the board at *every single step*. It recognizes that the board state at move 39 had a very high probability of winning, and assigns a massive penalty strictly to the 40th move where the expected value suddenly plummeted. 

### 4. Higher Data Efficiency & Faster Convergence
Is TD mathematically sound? Yes! For any fixed policy, TD(0) is proven to converge to the true value function $v_\pi$ (as long as the learning rate $\alpha$ is sufficiently small). 

But a more practical question is: *Which method learns faster?* 
Empirically, TD methods converge to the correct values using much less data than Monte Carlo methods. We can see this clearly in the **Random Walk** experiment.

**The Random Walk Setup:**
* An agent starts in a center state (C) and randomly steps left or right across 5 states (A, B, C, D, E).
* Terminating on the far right gives a reward of +1. Terminating on the far left gives 0.
* The true value of each state is simply the probability of terminating on the right (e.g., State C = 0.5).

![Random Walk Setup](images/RandomWalk-Setup.png.png)
*(Caption: The 5-state Random Walk environment. All episodes begin in state C.)*

To prove TD's efficiency, we look at the Root Mean-Squared (RMS) error of both algorithms over 100 episodes. The graph below (reproduced from Sutton & Barto, 2018, p. 150) highlights the performance difference:

![Random Walk RMS Error](images/RandomWalk-Gragh.png.png)
*(Caption: Learning curves comparing TD(0) and constant-$\alpha$ MC on the Random Walk task. Figure from Sutton & Barto, 2018.)*

**Interpreting the Graph:**
* The **dashed lines** (Monte Carlo) decrease slowly.
* The **solid lines** (TD) drop much faster, regardless of the specific learning rate ($\alpha$) chosen. 
Because TD bootstraps, it leverages the learned structure of the environment and reaches a highly accurate prediction much faster than MC ever could.

💡 **Why are the lines fluctuating?**
If you look closely at the solid TD lines, they don't perfectly flatten out; they continue to slightly bounce up and down. This happens because we are using a constant step-size parameter ($\alpha$). Even after getting close to the true value, the algorithm keeps adjusting its guess based on the most recent random episode. To stop the fluctuations, we would need to gradually decrease $\alpha$ over time.

---

## 5. Deep Dive: The Optimality of TD(0)

*(Critical Thinking: What does it mean for an algorithm to be "Optimal"?)*

We established that both TD and MC eventually converge to the correct predictions. But if we only have a limited, finite amount of data, how does each algorithm process it? 

To understand this, researchers use **Batch Updating**. This means we take our limited data (say, 8 episodes) and feed it to the algorithm over and over again until the value estimates stop changing. 

Batch MC and Batch TD both converge deterministically, but *they converge to two completely different answers*. Let's look at a famous thought experiment from Sutton & Barto (2018, p. 152) to understand why.

### Example: You are the Predictor
Place yourself in the role of the RL agent. You are observing an unknown Markov reward process, and you receive the following batch of 8 episodes:

* **Episode 1:** A $\rightarrow$ Reward 0 $\rightarrow$ B $\rightarrow$ Reward 0 $\rightarrow$ Terminate
* **Episodes 2-7:** B $\rightarrow$ Reward 1 $\rightarrow$ Terminate
* **Episode 8:** B $\rightarrow$ Reward 0 $\rightarrow$ Terminate

**The Question:** Given this data, what are the optimal values for State B and State A?

![Predictor Setup](images/Optimality.png.png)
*(Caption: A visualization of the transition data. We must determine the values of A and B based only on this limited experience.)*

#### The Obvious Answer: Value of B
Everyone, regardless of the algorithm used, agrees on the value of State B. We started in State B a total of 8 times. Six times it gave a reward of 1, and two times it gave a reward of 0. 
* $V(B) = \frac{6}{8} = \mathbf{75\%}$

#### The Tricky Answer: Value of A
We only saw State A *once*. It transitioned to B, and the immediate reward was 0. What is $V(A)$? Here is where MC and TD completely disagree:

**1. The Monte Carlo Answer: $V(A) = 0\%$**
Monte Carlo calculates optimality by minimizing the mean-squared error of the *training data*. In the actual data provided, the only time we ever saw A, the total return that followed it was exactly 0. Therefore, MC strictly fits the historical data and declares $V(A) = 0$.

**2. The Temporal-Difference Answer: $V(A) = 75\%$**
TD computes what is known as the **Maximum-Likelihood Estimate** (also called the **Certainty-Equivalence Estimate**). 
TD looks at the underlying Markov structure and says: "100% of the time, State A transitions to State B with zero reward. Therefore, the value of A is simply the value of B!" Since $V(B) = 75\%$, TD declares $V(A) = \mathbf{75\%}$.

### Which answer is better?
While Monte Carlo gives zero error on the *past* training data, the TD answer makes much more logical sense. 

If this environment is truly a Markov process, we expect the TD answer to produce much lower errors on **future, unseen data**. Monte Carlo simply memorizes the past, while TD actually builds a working model of the environment's dynamics in the background. This certainty-equivalence property is exactly why TD learns so much faster and is more data-efficient than MC.

---

## 6. Python & Library Suggestions

If you want to practice implementing TD Prediction, you do not need to build environments from scratch. The Python ecosystem has standard tools designed exactly for this.

**Recommended Libraries:**
* **Numpy:** For maintaining tabular value functions as multidimensional arrays or dictionaries.
* **[Gymnasium (Farama Foundation)](https://gymnasium.farama.org/):** The modern, maintained standard API for RL environments (formerly known as OpenAI Gym).

### Environments for TD(0)
To test if your TD(0) prediction algorithm works, you need environments where an agent follows a set policy, and your algorithm calculates the expected returns.

1. **`FrozenLake-v1`:** A simple grid world where transitions are slippery (stochastic). It is perfect for testing if your TD(0) algorithm correctly evaluates the danger of standing next to a hole.
2. **`CliffWalking-v0`:** A classic grid environment where stepping off the cliff yields a massive negative reward (-100). 

### Setup Code Snippet
Here is how you can easily initialize a Gymnasium environment to begin feeding data to the TD(0) update function we defined earlier:

```python
import gymnasium as gym
import numpy as np

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Initialize the Value table to zeros (State space is 16 for FrozenLake)
V = np.zeros(env.observation_space.n)

# Reset environment to get the starting state
state, info = env.reset()

# Example of taking a random step and getting the data needed for a TD update
action = env.action_space.sample() # A random policy
next_state, reward, terminated, truncated, info = env.step(action)

# Now you would pass these variables into your TD(0) update function!
# V = update_td_0(V, state, reward, next_state)
```

---

## 7. Conclusion

Temporal-Difference learning is arguably the most central and novel idea in Reinforcement Learning. 

By blending the raw experience sampling of **Monte Carlo** with the bootstrapping power of **Dynamic Programming**, TD(0) gives us an algorithm that learns efficiently, continuously, and online. As we saw in the Random Walk and "Predictor" examples, TD is vastly more data-efficient than Monte Carlo because it converges to the Certainty-Equivalence estimate—meaning it actually learns the underlying Markov structure of the world rather than just memorizing past outcomes.

However, in this section, we only solved the **Prediction Problem**. We learned how to find the value function $V(S)$ for a *fixed* set of rules. 

But what if we want the agent to learn the *best possible policy* while simultaneously exploring the environment? To do that, we must shift our focus from evaluating states $V(S)$ to evaluating actions $Q(S, A)$. This brings us to the exciting world of **On-Policy TD Control**, famously known as **SARSA**, which will be covered in the next section!

---

## 8. References

**References:**
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (Specifically Chapter 6: Temporal-Difference Learning, pp. 143-154).

