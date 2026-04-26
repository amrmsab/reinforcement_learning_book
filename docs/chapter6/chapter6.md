# Chapter 6: Function Approximation in Reinforcement Learning

## 1. Introduction: Why Function Approximation?

In earlier chapters, we assumed that value functions could be stored in **tables**, with one entry per **state–action pair**. While intuitive, this approach quickly becomes infeasible in real-world problems.

Consider:

- Continuous state spaces (e.g., position and velocity)
- High-dimensional observations (e.g., images)
- Large combinatorial environments

In such cases:

- The number of state–action pairs is effectively **infinite**, or
- The table becomes too large to **store, learn, or generalize from**

The fundamental challenge is **generalization**:

> How can an agent learn action values from limited experience and apply that knowledge to unseen state–action pairs?

Function approximation provides the answer. Instead of storing values explicitly, we learn a **parameterized action-value function**:

$$
\hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a)
$$

For linear function approximation, this is often written as
$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s, a)$$

Here, $\mathbf{w}^T$ is the transpose of the weight vector $\mathbf{w}$, and $\mathbf{x}(s, a)$ is the feature vector for the state–action pair $(s, a)$.

In general, we build $\mathbf{x}(s, a)$ by giving each action its own partition of the vector. The features for the current state are placed only in the block for the chosen action, and the other action blocks are set to zero.

For example, if there are two actions, $a_1$ and $a_2$, and three state features $[f_1, f_2, f_3]$, then:

$$
\mathbf{x}(s, a_1) = [f_1, f_2, f_3, 0, 0, 0]
$$

$$
\mathbf{x}(s, a_2) = [0, 0, 0, f_1, f_2, f_3]
$$

This shifts the problem from **memorization** to **learning a function**, enabling the agent to generalize across similar states and actions.

---

## 2. From Tables to Functions

In tabular methods:

```python
Q[s, a] = updated independently
```

With function approximation:

- A single update affects many states
- Knowledge is shared across similar inputs
- Learning becomes scalable

```python
w ← w + α * error * gradient
```

A single update now affects many states, allowing knowledge to generalize across the state space. This coupling is both the main advantage and the main challenge of function approximation.

This means that learning in one part of the state space influences many others, making the representation critical for controlling how knowledge is shared.

## 3. Feature Construction: Representing the State Space

The effectiveness of function approximation depends heavily on feature design. Features define how states are represented numerically.

We focus on four key methods to construct $\mathbf{x}(s, a)$:

### 3.1 Coarse Coding

Coarse coding represents states using overlapping regions.

Each feature corresponds to a region (e.g., circle in space)
A feature is 1 if the state lies in the region, otherwise 0

Nearby states share features, which enables generalization.

<figure>
	<img src="figures/Coarse%20Coding.png" alt="Coarse coding circles overlapping in 2D space">
	<figcaption>Coarse coding circles overlapping in 2D space. Two states share features, which creates partial generalization.</figcaption>
</figure>

**How to control generalization**

- **Region size**: Larger regions → broader generalization; smaller regions → more localized updates
- **Overlap amount**: More overlap → smoother transitions between states
- **Number of features/regions**: More features increase the representational capacity and allow finer distinctions

A useful way to think about coarse coding is that it defines _which states should influence each other during learning_.

<figure>
	<img src="figures/Coarse%20Coding%20Generalization.png" alt="Coarse coding generalization with different region sizes">
	<figcaption>Larger overlapping regions increase feature sharing across states, which improves generalization.</figcaption>
</figure>

**How it works:**

In practice, coarse coding defines a set of regions (receptive fields) that cover the state space, often with significant overlap. When a state is observed, multiple features may become active simultaneously. The approximate value (or action-value) is then computed as a weighted sum of all active features. Updating one state also updates nearby states that share features.

### 3.2 Tile Coding

Tile coding improves coarse coding by using structured grids:

Multiple overlapping grids (tilings)
Each tiling partitions space differently
Exactly one tile active per tiling

<figure>
	<img src="figures/Tile%20Coding.png" alt="Multiple offset grids for tile coding">
	<figcaption>Multiple offset grids. A state activates one tile in each tiling/grid, which creates a richer representation.</figcaption>
</figure>

**How to control generalization**

- **Tile width**: Larger tiles → broader generalization across states
- **Number of tilings**: More tilings → higher resolution and better discrimination
- **Offsets between tilings**: Proper offsets prevent identical partitions and improve representation quality

**How it works:**

Each tiling acts as an independent partition of the state space. For a given state, exactly one tile per tiling is active, meaning the total number of active features is fixed. The final feature vector is formed by combining all active tiles across tilings. This structured sparsity makes computation efficient and ensures stable learning updates.

Because different tilings are offset, nearby states will share _some but not all_ active tiles. This creates controlled partial generalization: similar states influence each other, but not uniformly.

Tile coding is especially powerful because it provides a **direct and interpretable way to tune generalization vs. precision**.

---

### 3.3 Radial Basis Functions (RBFs)

RBFs use smooth, continuous features:

$$
x_i(s) = \exp\left(-\frac{\|s - c_i\|^2}{2\sigma_i^2}\right)
$$

Here, $x(s)$ is the feature vector, and each component $x_i(s)$ measures how strongly state $s$ activates the $i$th radial basis function.

Here, $c_i$ is the center of the $i$th basis function, and $\sigma_i$ controls its width or spread. A larger $\sigma_i$ makes the basis function broader, so nearby states receive more similar feature values.

Intuitively, each center $c_i$ represents a **prototype state** or a reference point in the state space. The radial basis function measures how similar the current state $s$ is to this prototype. If $s$ is close to $c_i$, the feature $x_i(s)$ will be close to 1; if it is far away, the feature value will decay smoothly toward 0.

In this sense, $c_i$ defines _where_ in the state space a feature is focused, while $\sigma_i$ defines _how far its influence extends_. Together, they determine both the location and the spread of generalization.

<figure>
	<img src="figures/RBF.png" alt="Radial basis function features centered at different points">
	<figcaption>Radial basis functions centered at different points. States closer to a center activate that feature more strongly.</figcaption>
</figure>

- Each feature measures similarity to a center
- Produces smooth value functions

**How to control generalization**

- **Width parameter ($\sigma_i$)**:
  - Large $\sigma$ → broad, smooth generalization
  - Small $\sigma$ → localized, sharp responses
- **Number of centers**: More centers → higher representational capacity
- **Placement of centers ($c_i$)**:
  - Uniform grid → structured coverage
  - Data-driven placement → focuses on important regions

**How it works:**

Unlike coarse or tile coding, RBF features are **continuous-valued**, not binary. Each feature responds smoothly based on distance to its center. The closer a state is to a center, the higher the activation. The approximate value is computed as a weighted sum of these smooth activations, resulting in a continuous and differentiable function.

This smoothness allows the agent to generalize in a more natural way, especially in environments where value changes gradually across the state space.

RBFs provide fine-grained control over _how quickly similarity decays with distance_, making them well-suited for smooth environments.

---

### 3.4 Kanerva Coding

- Randomly chosen prototype states
- Features activated by similarity

**How it works:**

Kanerva coding represents states by comparing them to a set of randomly selected **prototype states**. Each feature corresponds to one prototype and becomes active if the current state is sufficiently similar to it (e.g., within a certain distance threshold). This creates a sparse binary feature representation.

Unlike grid-based methods, Kanerva coding does not rely on geometric partitioning. Instead, it distributes prototypes throughout the state space, often randomly, allowing it to scale to very high-dimensional problems.

**How to control generalization**

- **Number of prototypes**: More prototypes → richer representation
- **Similarity threshold**: Determines how many features activate for a given state
- **Distance metric**: Defines what “similar” means (e.g., Euclidean distance, Hamming distance)

A key advantage is that **complexity depends on the number of features, not the dimensionality of the state space**, making Kanerva coding particularly useful in high-dimensional settings.

### Comparison of Feature Methods

The methods above differ mainly in how they control generalization:

- Coarse coding: simple overlapping regions, intuitive but less structured
- Tile coding: structured and efficient, with precise control over resolution
- RBFs: smooth and continuous generalization, suitable for gradual value changes
- Kanerva coding: scalable to high-dimensional spaces using similarity to prototypes

Choosing a feature representation is often more important than the learning algorithm itself, as it determines how experience generalizes across the state space.

## 4. Control with Function Approximation

### 4.1 The Control Loop

We now combine function approximation with reinforcement learning to enable control in large or continuous environments.

At each time step:

1. Estimate action values:

$$
\hat{q}(s, a, \mathbf{w}) \quad \forall a
$$

2. Select an action using $\varepsilon$-greedy behavior:
   - With probability $\varepsilon$, explore
   - Otherwise, choose the best action
3. Observe the reward and next state
4. Update the weights using SARSA

### 4.2 SARSA with Function Approximation

Update rule:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[\underbrace{R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t)}_{U_t\,\text{(target)}} - \underbrace{\hat{q}(S_t, A_t, \mathbf{w}_t)}_{\text{estimate/prediction}}\right] \underbrace{\nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w}_t)}_{\text{gradient}}
$$

1. $\alpha$ is the learning rate.
2. $\gamma$ is the discount factor.
3. The TD error is

$$
\delta_t = U_t - \hat{q}(S_t, A_t, \mathbf{w}_t) = R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t).
$$

### Pseudocode for SARSA with function approximation

We first present the simpler one-step version of SARSA, often called SARSA(0), before extending it to the more general SARSA(λ) algorithm with eligibility traces.

#### 1. SARSA(0)

```text
Initialize parameters θ arbitrarily (e.g., θ = 0)

for each episode do
	Choose initial state s
	Choose a from s using policy derived from Q_θ (e.g., ε-greedy)
	repeat
		Execute action a, observe r, s'
		Choose a' from s' using policy derived from Q_θ (e.g., ε-greedy)
		θ ← θ + α (r + γ Q_θ(s', a') − Q_θ(s, a)) φ(s)
		s ← s'; a ← a'
	until s is terminal
end for
```

#### 2. SARSA(λ) with Eligibility Traces

```text
Initialize weight vector θ

For each episode:

    Initialize eligibility trace vector e = 0

    Observe initial state S
    Choose action A using ε-greedy policy from q_hat(S, A, θ)

    Repeat for each step:

        Take action A
        Observe reward R and next state S'

        q_current = q_hat(S, A, θ)

        Update eligibility trace:
            e = γ λ e + ∇q_hat(S, A, θ)

        If S' is terminal:
            δ = R - q_current
            θ = θ + α δ e
            End episode

        Else:
            Choose next action A' using ε-greedy policy from q_hat(S', A', θ)

            q_next = q_hat(S', A', θ)

            δ = R + γ q_next - q_current

            θ = θ + α δ e

            S = S'
            A = A'
```

### Interpreting the Algorithms

The two algorithms above share the same core idea: they update the parameters of an approximate action-value function using the **temporal-difference (TD) error**:

$$
\delta = r + \gamma \hat{q}(s', a') - \hat{q}(s, a)
$$

This error measures how “surprised” the agent is after taking an action. If the observed outcome is better than expected, the weights are increased; if it is worse, they are decreased.

---

### From SARSA(0) to SARSA(λ)

SARSA(0) performs a **local update**:

- Only the features associated with the current state–action pair $(s, a)$ are updated
- Learning is simple and computationally efficient
- However, credit assignment is limited to a single step

In contrast, SARSA(λ) introduces **eligibility traces**, which act as a short-term memory of recently visited state–action pairs.

Instead of updating only the current features, SARSA(λ):

- Accumulates traces for recently visited features
- Uses a single TD error to update **multiple past steps**
- Allows credit (or blame) to propagate backward in time

---

### Intuition Behind Eligibility Traces

The eligibility trace vector $e$ can be interpreted as a **decaying memory**:

- When a feature becomes active, its trace increases
- Over time, traces decay by a factor of $\gamma \lambda$
- Recently visited features have higher traces → receive larger updates

This mechanism enables the agent to answer the key question:

> Which past decisions contributed to the current outcome?

---

### The Role of $\lambda$

The parameter $\lambda \in [0,1]$ controls how far updates propagate:

- $\lambda = 0$ → equivalent to SARSA(0), only current step is updated
- $\lambda \approx 1$ → long-range credit assignment (similar to Monte Carlo)
- Intermediate values → balance between bias and variance

This creates a continuum between:

- **Bootstrapping methods** (low $\lambda$)
- **Full return methods** (high $\lambda$)

---

### Practical Considerations

- **SARSA(0)** is easier to implement and requires less computation per step
- **SARSA(λ)** typically learns faster in problems with delayed rewards
- The choice of $\lambda$ is task-dependent and often tuned empirically

When combined with function approximation, especially linear methods, SARSA(λ) provides a powerful and scalable approach for learning in continuous or high-dimensional environments.

---

### Key Takeaway

> SARSA(0) updates _what just happened_, while SARSA(λ) updates _what led to what just happened_.

This distinction becomes crucial in environments where rewards are delayed and correct behavior depends on sequences of actions rather than single decisions.

---

## 5. Bootstrapping: A Critical Design Choice

The use of SARSA and SARSA(λ) introduces an important concept in reinforcement learning: **bootstrapping**.

Bootstrapping means:

> Updating estimates using other learned estimates

In temporal-difference methods such as SARSA, this means that the target value is constructed using the current estimate of the value function itself. Instead of waiting until the end of an episode to compute the true return, the agent updates its estimate using another estimate of the next state–action pair:

$$
\delta = r + \gamma \hat{q}(s', a') - \hat{q}(s, a)
$$

This allows learning to occur incrementally, step by step, during interaction with the environment.

---

### 5.1 Bootstrapping vs Non-Bootstrapping

| Method            | Target                | Example     |
| ----------------- | --------------------- | ----------- |
| Bootstrapping     | Uses current estimate | TD, SARSA   |
| Non-bootstrapping | Uses full return      | Monte Carlo |

The key distinction is whether the target depends on the agent’s current estimate:

- In bootstrapping methods, the target includes $\hat{q}(s', a')$, which is itself an estimate
- In non-bootstrapping methods, the target is based only on actual observed rewards

This difference has important consequences for learning speed and stability.

---

### 5.2 Why Use Bootstrapping?

Despite weaker theoretical guarantees, bootstrapping methods are widely used in practice because they:

- **Learn faster** by updating after every step instead of waiting until the end of an episode
- **Use less data**, since each transition contributes immediately to learning
- **Work well with function approximation**, especially in large or continuous state spaces

However, bootstrapping introduces bias because it relies on imperfect estimates rather than true returns.

---

### 5.3 Intuition

Bootstrapping can be thought of as _learning from guesses_. The agent updates its value estimates using other estimates that may still be inaccurate. This introduces bias, but it significantly reduces variance and speeds up learning.

In contrast, non-bootstrapping methods (such as Monte Carlo) wait for the full outcome before updating. This provides unbiased targets, but can be slow and inefficient, especially in long episodes.

---

### 5.4 When to Use Each

**Use bootstrapping when:**

- Learning must happen online (step-by-step)
- Data is limited or expensive to collect
- The environment has long episodes or delayed rewards
- Function approximation is used

**Use non-bootstrapping when:**

- Accurate, unbiased estimates are more important than speed
- Episodes are short and complete returns are easy to obtain
- Training can bedone offline with full trajectories

---

### Key Takeaway

In practice, most modern reinforcement learning methods rely on bootstrapping because its efficiency and ability to learn online outweigh its theoretical limitations.

## 6. Case Study: The Mountain Car Problem

This problem highlights the importance of function approximation, as the continuous state space makes tabular methods infeasible. Using methods such as tile coding with SARSA allows the agent to generalize across states and learn an effective policy.

### 6.1 Problem Description

- State: position + velocity (continuous)
- Actions: accelerate left, right, or none
- Goal: reach the top of a hill

Challenge:

- Engine is too weak to climb directly
- Agent must first move away from the goal

### 6.2 Why This Is Difficult

- Requires long-term planning
- Immediate rewards are misleading
- State space is continuous

### 6.3 Learned Behavior

The agent learns to:

- Move left (away from goal)
- Build momentum
- Accelerate right to reach the goal

<figure>
  <img src="figures/Mountain%20Car%20Example.png" alt="Mountain Car Example">
  <figcaption>The mountain–car task (upper left panel) and the cost-to-go
function (−maxa ˆq(s,a,w)) learned during one run.</figcaption>
</figure>

## 7. Trade-offs in Function Approximation

While powerful, function approximation introduces new challenges:

### Strengths

- Scales to large/continuous spaces
- Enables generalization
- Works well with bootstrapping

### Limitations

- Sensitive to feature design
- May diverge (especially off-policy)
- Harder to analyze than tabular methods

Important takeaway:

> The success of RL with function approximation depends more on representation than on the algorithm itself.
