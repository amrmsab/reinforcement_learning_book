# Chapter 3: Finite Markov Decision Processes — Teaching a Machine to Think Ahead

> _"The world is not a slot machine."_
> 

---

## Before You Start Reading

This chapter builds on the ideas introduced in Chapter 2, where we explored how an agent can learn from feedback without knowing anything in advance — specifically through the _k-armed bandit_ problem. If you haven't read that yet, don't worry; we'll recap the key distinction right at the start. By the end of this chapter, you'll have a solid grasp of the mathematical backbone of reinforcement learning: **Finite Markov Decision Processes**, or **MDPs** for short.

No prior experience with probability theory beyond the basics is assumed. If you've seen a conditional probability expression before, you're in good shape. If not, we'll explain it as we go.

---

## 3.0 Introduction: The World Is Not a Slot Machine

Imagine you're playing a slot machine. You pull the lever, you get a reward (or not), and then you pull again. Each pull is independent. The machine doesn't change based on what you did before, and your reward today doesn't make tomorrow's rewards any different. This is the _k-armed bandit_ problem from Chapter 2 — clean, simple, memoryless.

Now imagine you're playing chess. You move a pawn. That single move doesn't just produce an immediate outcome; it reshapes the entire board. Your future options are now different. The opponent responds. The game is a sequence of decisions, each one building on the last, each one shaping what comes next.

That's the world MDPs are built to describe. They extend the bandit framework in one crucial direction: **your actions have consequences that ripple through time**. In the bandit world, you evaluate each action in isolation. In the MDP world, you have to think about _what comes next_, and what comes after that, all the way to the end — or forever, if there is no end.

This is why MDPs are the central formalism for reinforcement learning. They are, at their core, a mathematical language for sequential decision-making under uncertainty.

> **Key distinction to hold onto:** In the k-armed bandit problem, each action only affects your immediate reward. In an MDP, each action affects your _immediate reward_ **and** your _future situation_, and through that situation, all your future rewards. This single difference changes everything.

---

## 3.1 The Agent–Environment Interface: Who Does What?

Every MDP involves two characters:

- **The agent** — the learner and decision-maker. Think of it as the "brain."
- **The environment** — everything outside the agent. Think of it as the "world."

These two are locked in a continuous dialogue. At every moment, the environment tells the agent what's going on, the agent decides what to do, and then the environment responds. Rinse and repeat — forever, or until the task ends.

### 3.1.1 The Interaction Cycle

Here's what that dialogue looks like, written out as a sequence of time steps:

At time step $t$:

1. The environment presents the agent with a **state** $S_t$ — a snapshot of the current situation.
2. The agent looks at $S_t$ and picks an **action** $A_t$.
3. One step later, the environment responds with two things: a **reward** $R_{t+1}$ (a number reflecting how good that action was), and a **new state** $S_{t+1}$.

And then the cycle repeats. Over time, this produces a long sequence — called a **trajectory** — that looks like this:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots$$

<p align="center"> <img src="assets/mdp-cycle.png" width="100%" alt="Agent-environment interaction cycle diagram"> </p><p align="center"><em><strong>Figure 3.2</strong> — The agent–environment interaction cycle.</em></p>

It's worth pausing on the subscripts. Notice that the reward at time step $t+1$ (written $R_{t+1}$) arrives _after_ the action $A_t$. This notation emphasizes that the reward is a consequence of the action, not something that existed before it.

**An analogy:** Think of the agent as a driver and the environment as the road. The state $S_t$ is what the driver sees out the windshield right now: the curve ahead, the speed, the other cars. The action $A_t$ is what the driver does: steer left, hit the brakes, accelerate. The reward $R_{t+1}$ is the outcome: did they stay on the road? Did they get closer to the destination? And the new state $S_{t+1}$ is what the windshield shows next. The driver's job — the agent's job — is to choose actions that lead to the best possible journey, not just the best next moment.

<p align="center"> <img src="images/fig_3_3_driving_analogy.png" width="720" alt="Driving analogy mapped to MDP state, action, and reward"> </p> <p align="center"><em><strong>Figure 3.3</strong> — The driving analogy mapped onto MDP vocabulary. Show a car on a winding road annotated with three callout bubbles: (1) <strong>State S<sub>t</sub></strong> — the driver's view through the windshield (road curvature, speed gauge, traffic); (2) <strong>Action A<sub>t</sub></strong> — hands on the wheel, foot on a pedal; (3) <strong>Reward R<sub>t+1</sub> + Next State S<sub>t+1</sub></strong> — one moment later, with a ✓/✗ indicator showing whether things went well. The image should feel warm and human, anchoring the abstract MDP loop in something every reader has experienced.</em></p>

### 3.1.2 The Dynamics Function: Writing the Rules of the World

Now, how exactly does the environment produce the next state and reward from the current state and action? The answer is captured by a single mathematical object called the **dynamics function**, written as:

$$p(s', r \mid s, a) \doteq \Pr({S_t = s',\ R_t = r \mid S_{t-1} = s,\ A_{t-1} = a})$$

Let's unpack this. The function $p$ takes four arguments:

- $s$ — the current state (where you are now)
- $a$ — the action taken
- $s'$ — the next state (where you end up)
- $r$ — the reward received

And it returns a **probability**: how likely is it that you end up in state $s'$ with reward $r$, given that you were in state $s$ and took action $a$


In a **finite** MDP, the sets of states, actions, and rewards are all finite, there are a countable number of possibilities for each. This is what the word "finite" in the chapter title refers to. It's a simplifying assumption that makes the mathematics tractable, and it covers a surprisingly wide range of real-world problems.

**One important constraint**: for any given state $s$ and action $a$, the probabilities across all possible next states and rewards must sum to 1:

$$\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1, \quad \text{for all } s \in \mathcal{S},\ a \in \mathcal{A}(s)$$

This is just saying: _something_ must happen next. The environment can't freeze.

> The dynamics function $p$ is the complete description of the environment. If you know $p$, you know everything about how the world works.

### 3.1.3 The Markov Property: Forgetting the Past (on Purpose)

There's a crucial assumption hiding inside the dynamics function. Notice that $p(s', r \mid s, a)$ only conditions on the _current_ state $s$ and action $a$. It doesn't look at the whole history of states and actions before $s$. This is the **Markov property**:

> **The future is independent of the past, given the present.**

More formally: the current state $S_t$ must contain all the information from the history $S_0, A_0, R_1, \ldots, S_{t-1}, A_{t-1}$ that is relevant for predicting the future. If the state representation satisfies this, it is called a **Markov state**, and the process itself is a Markov Decision Process.

This sounds like a strong assumption — and it is. But it's often satisfied in practice, _as long as you define the state carefully_. The state doesn't have to be tiny. A chess board position, for instance, is a valid Markov state: knowing the current arrangement of pieces is enough to determine all future possibilities, without needing to remember every move that led there.

---

## 3.2 Goals and Rewards: What Are We Actually Optimizing?

We've established that the agent receives rewards. But what exactly is the agent trying to do with them?

The answer is formalized in what Sutton and Barto call the **Reward Hypothesis**:

> **All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).**

This is a big claim. It says that any goal: winning a game, driving safely, managing a portfolio, all can be encoded as a reward signal that the agent should try to maximize over time.

The philosophy here is important: **the reward signal communicates _what_ you want the agent to achieve, not _how_ you want it to achieve it.** If you want an agent to win at chess, you reward it for winning the game, not for taking the opponent's pieces, not for controlling the center, not for any intermediate tactic. The moment you start rewarding intermediate steps, you risk the agent finding clever ways to rack up those sub-rewards while completely losing the forest for the trees.

**A real example of this going wrong:** In the early days of RL research, a simulated boat racing agent was rewarded for its score in a circuit race. Instead of finishing the race, it discovered that spinning in circles on a patch of power-up tiles gave it a higher score than actually racing. The reward was technically being maximized — just not in the way the designers intended [Amodei et al., 2016]. This is called **reward hacking**, and it's one of the central challenges of modern AI safety.

---

## 3.3 Returns and Episodes

### 3.3.1 Defining the Return

If the agent's goal is to maximize cumulative reward, we need a formal way to write "cumulative reward." We call this the **return**, denoted $G_t$:

$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T$$

where $T$ is the final time step. The return is simply the sum of all rewards the agent collects from time $t$ onward. The agent's goal is to choose its actions so that $\mathbb{E}[G_t]$ — the _expected_ return — is as large as possible.

### 3.3.2 Episodic vs. Continuing Tasks

The formula above assumes there's a finite final time step $T$. This is true for many tasks: a chess game ends when someone is checkmated, a maze run ends when the agent reaches the exit, a video game episode ends when the player wins or loses. These are called **episodic tasks**.

In an episodic task, the agent's experience naturally breaks into chunks — _episodes_ — each one starting fresh. Crucially, each new episode begins independently of how the previous one ended: a new chess game can start regardless of whether you won or lost the last one.

But many tasks don't have a natural ending. A temperature control system for a server farm runs continuously. A stock trading algorithm operates day after day with no defined finale. A robot doing inventory in a warehouse keeps going as long as the warehouse does. These are **continuing tasks**, and they create a mathematical problem: if $T = \infty$, the simple sum could grow without bound, making $G_t = \infty$ — which is not a useful thing to optimize.

### 3.3.3 Discounting: Valuing the Future Less (but Not Ignoring It)

The solution to the infinite-sum problem is **discounting**. Instead of treating all future rewards equally, we give rewards in the near future more weight than rewards in the distant future. We introduce a parameter $\gamma \in [0, 1)$ called the **discount rate**, and redefine the return as:

$$G_t \doteq \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

The factor $\gamma^k$ means that a reward $k$ steps in the future is worth only $\gamma^k$ as much as an immediate reward. As long as $\gamma < 1$ and the rewards are bounded, this infinite sum converges to a finite value.

Two extreme cases:

- **$\gamma = 0$**: The agent is completely _myopic_ — it only cares about the very next reward. It has no concept of tomorrow.
- **$\gamma \to 1$**: The agent is fully _farsighted_ — it values the distant future almost as much as the present. It's playing the long game.

<p align="center"> <img src="assets/gamma-decay.jpg" width="100%" alt="Exponential decay curves for three values of the discount factor gamma"> </p><p align="center"><em><strong>Figure 3.6</strong> — How γ shrinks the effective weight of future rewards.</em></p>

In practice, $\gamma$ is a design choice. A value like $\gamma = 0.99$ gives a nice blend: the agent cares deeply about the future, but rewards very far away (say, 1000 steps) are effectively discounted to near-zero importance.

**Why discounting is intuitive:** Think about money. Would you rather receive £100 today or £100 in five years? Almost everyone prefers the money now — partly because of uncertainty (will the future reward actually arrive?), and partly because you can do something useful with money now that you can't if you wait. Discounting in RL captures exactly this intuition mathematically.

There's also a very clean recursive relationship worth knowing. The return at time $t$ can be written in terms of the return at time $t+1$:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

This one-liner is deceptively powerful. It says: the value of being in a state right now is the immediate reward plus a discounted version of all future values. This recursive structure is what makes dynamic programming algorithms (like Value Iteration and Policy Iteration, coming in later chapters) computationally feasible.

Notice how a myopic agent (low $\gamma$) barely values that final reward, while a farsighted agent (high $\gamma$) sees nearly its full value even three steps away.

---

## 3.4 Unifying the Notation: One Formula to Rule Them All

At this point we have two separate formulas: one for episodic tasks (finite sum up to $T$) and one for continuing tasks (infinite discounted sum). It would be convenient if we could handle both with a single framework.

We can. The trick is to think of episodic task termination as entering a special **absorbing state** — a fictional state that transitions only to itself and always produces a reward of zero. Once you're in it, you're stuck there forever, collecting nothing.

<p align="center"> <img src="assets/terminal-state.png" width="100%" alt="State-transition chain ending in an absorbing terminal state with a self-loop labelled R=0"> </p><p align="center"><em><strong>Figure 3.7</strong> — The absorbing terminal state trick.</em></p>

Mathematically, this means we can always write the return as:

$$G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$$

with the understanding that:

- For **episodic tasks**, $T$ is finite and $\gamma$ can be 1 (no discounting needed, since the episode ends).
- For **continuing tasks**, $T = \infty$ and $\gamma < 1$ (discounting needed to keep the sum finite).
- But never both $T = \infty$ and $\gamma = 1$ at the same time — that's the one forbidden combination, because the sum would be infinite.

This unified notation lets us write algorithms and proofs that work for both task types simultaneously, without having to maintain two separate cases throughout.

---

## 3.5 Putting It All Together: A Small Example

Let's walk through a complete, concrete MDP to make all these pieces click.

**Setup:** A robot is navigating a 3-cell corridor: cells A, B, and C. Cell C is the goal (reward: +10). The robot can move left or right. If it tries to move past the left wall from A, it stays in A. If it reaches C, the episode ends.

<p align="center"> <img src="images/fig_3_8_corridor_mdp.png" width="720" alt="3-cell corridor MDP with two coloured paths and annotated discounted returns"> </p> <p align="center"><em><strong>Figure 3.8</strong> — The 3-cell corridor MDP. Three cells in a row: <strong>A</strong> (start), <strong>B</strong> (neutral), <strong>C</strong> (goal — shade green or add a ★). Wall on the left of A. Labelled transition arrows: A→B ("right, r = 0"), B→A ("left, r = 0"), B→C ("right, r = +10"), and a self-loop at A for the blocked left move. Two overlaid paths: a <strong>green path</strong> A→B→C annotated "G<sub>0</sub> = 9" and an <strong>orange path</strong> A→B→A→B→C (dithering) annotated "G<sub>0</sub> = 7.29 (γ = 0.9)". One diagram — states, actions, rewards, transitions, and discounting all at once.</em></p>

|State|Action|Next State|Reward|
|---|---|---|---|
|A|right|B|0|
|A|left|A|0|
|B|right|C|+10|
|B|left|A|0|
|C|—|C|0|

With $\gamma = 0.9$ and the agent starting in A, if it goes right twice (A → B → C), the return is:

$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 = 0 + 0.9 \times 10 + 0 = 9$$

But if it dithers — going left from B before going right (A → B → A → B → C) — the return is:

$$G_0 = 0 + 0 \cdot \gamma + 0 \cdot \gamma^2 + 10 \cdot \gamma^3 = 10 \times 0.729 = 7.29$$

The discount factor "punishes" the inefficient path — not harshly, but enough to encourage the agent to find shorter routes. This is how discounting shapes behavior, even in a setting this simple.

---

## 3.6 Broader Connections and Critical Reflections

### The MDP Framework Is Powerful — but Not Neutral

The MDP formalism is elegant, but it carries hidden assumptions worth examining. When we say "the agent tries to maximize reward," we are implicitly saying that whoever designs the reward function has already decided what matters. In real applications, this is a deeply human, and potentially flawed, act.

**Ethical implications:** In healthcare, if an RL agent is rewarded for reducing the length of a patient's hospital stay, it might learn to discharge patients prematurely. If a hiring algorithm is rewarded for "efficiency," it might learn to penalize candidates from historically underrepresented groups because past data reflects biased outcomes. The MDP framework is a tool; like all tools, its consequences depend on who wields it and how.

**The partial observability problem:** Our framework assumes the agent has access to a true Markov state. In the real world, this is rarely guaranteed. A self-driving car's sensors might be obscured by rain; a poker player can't see the opponent's cards. When the state is only _partially_ observable, we enter the territory of Partially Observable MDPs (POMDPs) — a significantly harder problem that Chapter 17 of Sutton & Barto explores in depth.

**Scalability:** Finite MDPs with small state and action spaces are tractable. But real problems — like controlling a robot with continuous joint angles, or playing a video game from raw pixels — have enormous or continuous state spaces. The bulk of modern reinforcement learning research is about extending MDP ideas to these settings using function approximation (neural networks, mostly), which we'll encounter starting in Chapter 9.

---

## Summary and Key Takeaways

Let's consolidate what we've covered in this chapter.

**Core Terminology:** An MDP is a formal model of sequential decision-making. At each time step, the agent observes a state $S_t$, takes an action $A_t$, and receives a reward $R_{t+1}$ and new state $S_{t+1}$. The resulting trajectory is the raw material the agent learns from.

**Environment Dynamics:** The function $p(s', r \mid s, a)$ completely describes how the environment works — the probability of landing in state $s'$ with reward $r$, given state $s$ and action $a$. In a finite MDP, this function is fully tabulated.

**Goals via Rewards:** The agent's goal is to maximize the _expected return_ — the cumulative sum of rewards over time. The reward signal encodes _what_ we want, not _how_ to achieve it. Designing a good reward function is as much an art as a science.

**Episodic vs. Continuing Tasks:** Some tasks end (episodic); some run forever (continuing). Discounting (via $\gamma$) handles the continuing case by ensuring the return remains finite and by capturing the intuitive preference for sooner rewards over later ones.

**Unified Notation:** Both task types can be handled with the single formula $G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$, either with finite $T$ and $\gamma = 1$, or infinite $T$ and $\gamma < 1$.

--- 
## References

- **Sutton & Barto (2018)** — _Reinforcement Learning: An Introduction_ (2nd ed.), Chapter 3.
- **Amodei et al. (2016)** — _Concrete Problems in AI Safety_.

---
## Citation

To cite this chapter, please use the following BibTeX:

```bibtex
@misc{hawater_2026_ReinforcementLearning,
  author       = {Ahmed Mohamed Hawater},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter 3},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}
```