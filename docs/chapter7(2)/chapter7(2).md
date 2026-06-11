# Chapter 7 Part 2: Actor-Critic Methods — Learning with a Critic

**Author:** [Ahmed Said Saleh](https://www.linkedin.com/in/ahmedsaidsaleh/)

> _The previous chapter showed how to learn a policy directly. This chapter shows how to give that policy a critic, so the agent can learn from shorter, more frequent, and more stable feedback._

---

## Before You Start Reading

This chapter continues directly from the previous chapter on **[Policy Gradient Methods][policy-gradient]**. There, we moved away from value-based control methods such as **[Q-learning][q-learning]**, which first learn action values and then extract a policy. Instead, we learned to parameterise the policy directly:

$$
\pi_\theta(a \mid s),
$$

and to improve it by following the gradient of expected return.

> **Reading links.** If the formal agent-environment setup feels rusty, review **[Finite Markov Decision Processes][finite-mdp]**. If value-based control or bootstrapping feels unfamiliar, review **[Q-learning][q-learning]** and **[Temporal-Difference Learning][td-learning]**. If the policy-gradient update, REINFORCE, baselines, or advantage estimates feel unfamiliar, review **[Policy Gradient Methods][policy-gradient]** before continuing. After this chapter, the natural next step is **[Introduction to Deep Reinforcement Learning][deep-rl]**.

The previous chapter also introduced three ideas that are essential here:

1. **[REINFORCE][policy-gradient]** updates the policy using full returns $G_t$.
2. **[Baselines][policy-gradient]** reduce variance by comparing the return to what was expected.
3. **[Advantage][policy-gradient]** asks whether an action was better or worse than the usual action in that state:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

Actor-Critic methods begin exactly at that point. Instead of waiting until the end of an episode and using the noisy return $G_t$, we train a second learner to estimate value. This second learner becomes the **critic**. The policy being improved is the **actor**.

So the main question of this chapter is:

> How can a learned value estimate give the policy a better learning signal than raw Monte Carlo returns?

By the end of the chapter, you should be able to explain:

- why Actor-Critic methods are a natural continuation of policy gradients;
- what the actor and critic each learn;
- how the temporal-difference error acts like a one-step advantage estimate;
- how Actor-Critic reduces variance but introduces bias;
- how the 1D continuous-control example works step by step;
- why the linear actor and critic in the worked example are only pedagogical choices, and what scaling limitations remain before **[Deep Reinforcement Learning][deep-rl]**.

This chapter deliberately stays focused on the **core Actor-Critic idea**. Actor-Critic itself does not require the actor or critic to be linear: they can be neural networks. However, we postpone deep neural-network details to **[the next chapter][deep-rl]**, which explains function approximation, moving targets, target networks, and experience replay.

---

## 7.1 From REINFORCE to Actor-Critic

Recall the **[policy-gradient update from the previous chapter][policy-gradient]**:

$$
\theta \leftarrow \theta + \alpha\,G_t\,\nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

This says:

- if the return after action $a_t$ is high, make that action more likely;
- if the return is low, make that action less likely.

This is elegant, but it has one major weakness: **the return $G_t$ can be very noisy**.

Suppose an agent takes a good action early in an episode, but many random bad events happen later. REINFORCE may punish the early action even though it was not responsible for the poor final outcome. Conversely, a poor early action might be rewarded just because the rest of the episode went well by chance.

A baseline improves this:

$$
\theta \leftarrow \theta + \alpha\,(G_t - V(s_t))\,\nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

Now the update does not ask: “Was the return good in absolute terms?”

It asks: “Was the return better than expected from this state?”

That difference is the advantage estimate:

$$
\hat{A}_t = G_t - V(s_t).
$$

The problem is that $G_t$ still requires future rewards. If the episode is long, the signal is delayed. If the task is continuing and has no natural end, waiting for a full return is not even possible.

Actor-Critic replaces the full return with a **bootstrapped estimate**, the same central idea behind **[Temporal-Difference Learning][td-learning]**. Instead of waiting until the end, the critic estimates what will happen next.

---

## 7.2 The Critic as a Learned Baseline

The critic usually learns a value function:

$$
V_w(s) \approx V^\pi(s),
$$

where $w$ are the critic parameters.

This value function estimates the expected future return from state $s$ under the current policy. In plain language:

> The critic tries to answer: “From this state, how much reward do I expect the actor to collect?”

The actor still learns the policy:

$$
\pi_\theta(a\mid s),
$$

where $\theta$ are the policy parameters.

The actor asks:

> “What action should I take?”

The critic asks:

> “Was that action better or worse than expected?”

The critic does not directly choose actions. It evaluates the actor’s choices and turns experience into a training signal.

This changes the learning loop from a pure policy-gradient method into a two-part system:

![Actor-Critic Algorithm](images/actor_critic_loop.png)


The actor and critic learn together, but they solve different problems.

---

## 7.3 Temporal-Difference Error: A One-Step Advantage

The key learning signal in basic Actor-Critic is the **[temporal-difference error][td-learning]**, or TD error:

$$
\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t).
$$

This compares two quantities:

1. What the critic expected before the action:

$$
V_w(s_t)
$$

2. What the critic now estimates after seeing the reward and next state:

$$
r_t + \gamma V_w(s_{t+1})
$$

If the second quantity is larger, the action led to a better-than-expected situation. If it is smaller, the action led to a worse-than-expected situation.

So the TD error behaves like an immediate advantage estimate:

$$
\delta_t \approx A^\pi(s_t,a_t).
$$

This is the central bridge from **[policy gradients][policy-gradient]** to Actor-Critic.

The actor update becomes:

$$
\theta \leftarrow \theta + \alpha_\theta\,\delta_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

The critic update becomes a value-learning step:

$$
w \leftarrow w + \alpha_w\,\delta_t\,\nabla_w V_w(s_t).
$$

The same TD error trains both components:

- for the **actor**, $\delta_t$ says whether to increase or decrease the probability of the action;
- for the **critic**, $\delta_t$ says how wrong the value prediction was.

### Interpreting the sign of $\delta_t$

| TD error | Meaning | Actor response | Critic response |
|---|---|---|---|
| $\delta_t > 0$ | Outcome was better than expected | Make $a_t$ more likely in $s_t$ | Increase $V_w(s_t)$ |
| $\delta_t < 0$ | Outcome was worse than expected | Make $a_t$ less likely in $s_t$ | Decrease $V_w(s_t)$ |
| $\delta_t \approx 0$ | Outcome matched expectation | Small policy change | Small value correction |

This is why Actor-Critic can learn step by step instead of waiting for a full episode.

---

## 7.4 Bias and Variance: The Main Tradeoff

Actor-Critic is not simply “better REINFORCE.” It changes the type of error the agent makes.

**[REINFORCE][policy-gradient]** uses Monte Carlo returns. These are usually **unbiased**: if we average enough complete episodes, the estimate approaches the true expected return. But Monte Carlo returns can have very high variance.

Actor-Critic uses **[bootstrapping][td-learning]**. The TD target:

$$
r_t + \gamma V_w(s_{t+1})
$$

contains the critic’s own estimate. This usually has lower variance because it uses only one observed reward plus a value estimate. But it can be biased if the critic is wrong.

So Actor-Critic trades:

| Method | Learning signal | Main strength | Main weakness |
|---|---|---|---|
| REINFORCE | Full return $G_t$ | Unbiased in expectation | High variance and delayed feedback |
| REINFORCE with baseline | $G_t - V(s_t)$ | Lower variance | Still waits for future returns |
| Actor-Critic | TD error $\delta_t$ | Online, lower variance, faster feedback | Biased if critic is inaccurate |

This tradeoff is not a minor detail. It is one of the central design decisions in reinforcement learning.

Actor-Critic works well when the critic is accurate enough to provide useful guidance. It can fail when the critic becomes confidently wrong. In that case, the actor may learn from a misleading teacher.

A helpful analogy is a student learning with a coach:

- Without a coach, the student waits until the final exam to know whether their study strategy worked.
- With a good coach, the student gets immediate corrections after each practice session.
- With a bad coach, the feedback arrives quickly but points in the wrong direction.

Actor-Critic is powerful because it learns from immediate feedback, but its quality depends on the critic.

### What Actor-Critic Adds Beyond Monte Carlo and TD Methods

It is useful to compare Actor-Critic with the two families it combines.

**Compared with ordinary Monte Carlo methods**, Actor-Critic does not need to wait until an episode finishes before learning. Monte Carlo methods estimate value from complete returns such as:

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
$$

That is conceptually clean, but it creates two practical problems. First, the update is delayed. Second, the return may contain many random future events that the current action did not really cause. Actor-Critic replaces this long return with the shorter TD signal:

$$
\delta_t=r_t+\gamma V_w(s_{t+1})-V_w(s_t).
$$

This gives Actor-Critic three advantages over normal Monte Carlo learning:

- it can update after every transition, not only after full episodes;
- it usually has lower variance because the learning signal depends on one observed reward and one value estimate;
- it works naturally in continuing tasks where there may be no final episode return to wait for.

The cost is that Actor-Critic uses a learned estimate inside the target, so the update can be biased when the critic is wrong.

**Compared with pure TD learning**, Actor-Critic does more than estimate values. TD learning is excellent at prediction: it learns how good states or state-action pairs are by bootstrapping from later estimates. However, a TD value learner does not by itself directly parameterise a policy. In value-based control, the policy is usually extracted indirectly, for example by choosing:

$$
a=\arg\max_a Q(s,a).
$$

This becomes awkward when the action space is continuous, because there may be infinitely many possible actions to search over. Actor-Critic avoids this problem by giving the agent an explicit actor:

$$
\pi_\theta(a\mid s).
$$

The critic still uses the TD idea, but the actor learns the policy directly. This gives Actor-Critic several advantages over using TD learning alone:

- it can learn stochastic policies instead of only greedy value-based policies;
- it can represent continuous actions without enumerating every action;
- it separates **evaluation** from **control**, so the critic judges while the actor improves behaviour;
- it provides a direct bridge from value learning to policy optimization.

A compact way to say this is: Monte Carlo gives Actor-Critic the policy-gradient objective, TD gives it step-by-step bootstrapping, and the actor-critic architecture combines both into one learning loop.

---

## 7.5 Actor-Critic in Continuous Action Spaces

One reason Actor-Critic methods are important is that they naturally support **continuous actions**.

In many real control problems, the action is not a small set like:

$$
\{\text{left},\text{right},\text{up},\text{down}\}.
$$

Instead, the action may be a real number or vector:

- acceleration of a car;
- steering angle;
- motor torque;
- robot joint velocity;
- amount of power sent to a machine.

For continuous control, the actor often represents a probability distribution over actions. A common choice is a Gaussian policy:

$$
a \sim \mathcal{N}(\mu_\theta(s),\sigma^2).
$$

Here the policy does not output one action directly. It outputs the mean of a distribution, and the action is sampled from that distribution.

The mean $\mu_\theta(s)$ says where the policy is currently aiming. The standard deviation $\sigma$ controls exploration. A large $\sigma$ means the agent tries many different actions. A small $\sigma$ means the agent behaves more consistently.

The policy-gradient term:

$$
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

still works. It tells us how to change the parameters so that the sampled action becomes more or less likely.

If $\delta_t > 0$, the sampled action was better than expected, so the actor shifts the policy toward it.

If $\delta_t < 0$, the sampled action was worse than expected, so the actor shifts the policy away from it.

This makes Actor-Critic especially natural for control tasks.

---

## 7.6 Worked Example: A Car Moving on a 1D Track

Now let us use a simple example to see the mechanics clearly.

The example is intentionally small. Its goal is not to represent a full driving system. Its goal is to make the actor update, critic update, TD error, and continuous-action policy visible in numbers.

### Environment

A car moves along a one-dimensional track.

The state is:

$$
s=(x,v),
$$

where:

- $x$ is position;
- $v$ is velocity.

The car starts at:

$$
x=0,\quad v=1.
$$

The goal is to reach:

$$
x=10.
$$

The action is continuous acceleration:

$$
a\in\mathbb{R}.
$$

The dynamics are:

$$
v' = v + a,
$$

$$
x' = x + v'.
$$

The reward encourages reaching the target while avoiding excessive acceleration:

$$
r_t=-|x_{t+1}-10|-0.1a_t^2.
$$

The first term rewards being close to the target. The second term penalizes aggressive control.

### Actor

The actor uses a Gaussian policy:

$$
a \sim \mathcal{N}(\mu_\theta(s),\sigma^2),
$$

with a linear mean:

$$
\mu_\theta(s)=\theta_1x+\theta_2v.
$$

Initial actor parameters:

$$
\theta=(0.5,1.0).
$$

For simplicity:

$$
\sigma=1.
$$

### Critic

The critic uses a linear value function:

$$
V_w(s)=w_1x+w_2v.
$$

Initial critic parameters:

$$
w=(1.0,0.5).
$$

We use:

$$
\alpha=0.1,
$$

and for this short episodic calculation:

$$
\gamma=1.
$$

Using $\gamma=1$ is acceptable here because the demonstration is a finite, short-horizon calculation. In continuing tasks, $\gamma<1$ is normally required to keep returns finite.

---

## 7.7 First Transition: A Poor Action

### Step 1: Observe the state

The initial state is:

$$
s_0=(0,1).
$$

### Step 2: Sample an action

The actor computes the Gaussian mean:

$$
\mu(s_0)=0.5(0)+1.0(1)=1.
$$

Suppose the sampled action is:

$$
a_0=2.
$$

The action is larger than the current mean. The actor is exploring.

### Step 3: Apply the environment dynamics

Velocity becomes:

$$
v_1=1+2=3.
$$

Position becomes:

$$
x_1=0+3=3.
$$

So the next state is:

$$
s_1=(3,3).
$$

The reward is:

$$
r_0=-|3-10|-0.1(2^2)=-7-0.4=-7.4.
$$

### Step 4: Compute the TD error

The critic’s old estimate is:

$$
V(s_0)=1(0)+0.5(1)=0.5.
$$

The critic’s estimate of the next state is:

$$
V(s_1)=1(3)+0.5(3)=4.5.
$$

The TD error is:

$$
\delta_0=r_0+\gamma V(s_1)-V(s_0).
$$

Substituting values:

$$
\delta_0=-7.4+4.5-0.5=-3.4.
$$

The negative TD error means the result was worse than expected.

### Step 5: Update the actor

For a Gaussian policy with fixed variance, the score term is proportional to:

$$
\nabla_\theta\log\pi_\theta(a\mid s)=\frac{a-\mu(s)}{\sigma^2}[x,v].
$$

Here:

$$
(a_0-\mu(s_0))[x,v]=(2-1)(0,1)=(0,1).
$$

The actor update is:

$$
\Delta\theta=\alpha\delta_0(0,1).
$$

So:

$$
\Delta\theta=0.1(-3.4)(0,1)=(0,-0.34).
$$

Updated actor parameters:

$$
\theta=(0.5,0.66).
$$

The actor has reduced the influence of velocity on the action mean. Intuitively, because a large acceleration from this state produced a worse-than-expected result, similar actions become less likely.

### Step 6: Update the critic

The critic update is:

$$
\Delta w=\alpha\delta_0[x,v].
$$

Using $s_0=(0,1)$:

$$
\Delta w=0.1(-3.4)(0,1)=(0,-0.34).
$$

Updated critic parameters:

$$
w=(1.0,0.16).
$$

The critic has also reduced its value estimate for states where velocity contributes strongly.

---

## 7.8 Second Transition: A Better-Than-Expected Action

Now the current state is:

$$
s_1=(3,3).
$$

### Step 1: Compute the policy mean

Using the updated actor parameters:

$$
\mu(s_1)=0.5(3)+0.66(3)=1.5+1.98=3.48.
$$

Suppose the sampled action is:

$$
a_1=3.5.
$$

This is very close to the policy mean.

### Step 2: Apply the environment dynamics

Velocity becomes:

$$
v_2=3+3.5=6.5.
$$

Position becomes:

$$
x_2=3+6.5=9.5.
$$

So:

$$
s_2=(9.5,6.5).
$$

The reward is:

$$
r_1=-|9.5-10|-0.1(3.5^2).
$$

$$
r_1=-0.5-1.225=-1.725.
$$

This reward is still negative, but it is much better than before because the car is now close to the target.

### Step 3: Compute the TD error

Using the updated critic:

$$
V(s_1)=1(3)+0.16(3)=3.48.
$$

$$
V(s_2)=1(9.5)+0.16(6.5)=10.54.
$$

So:

$$
\delta_1=-1.725+10.54-3.48=5.335.
$$

This time the TD error is positive. The action led to a state that was much better than expected.

### Step 4: Update the actor

The score term is:

$$
(a_1-\mu(s_1))[x,v]=(3.5-3.48)(3,3).
$$

$$
=0.02(3,3)=(0.06,0.06).
$$

The actor update is:

$$
\Delta\theta=0.1(5.335)(0.06,0.06).
$$

$$
\Delta\theta\approx(0.032,0.032).
$$

Updated actor parameters:

$$
\theta=(0.532,0.692).
$$

The actor now slightly reinforces this behaviour because it performed better than the critic expected.

### Step 5: Update the critic

The critic update is:

$$
\Delta w=0.1(5.335)(3,3).
$$

$$
\Delta w\approx(1.6005,1.6005).
$$

Updated critic parameters:

$$
w=(2.6005,1.7605).
$$

The critic has strongly increased its value estimate for states like $s_1$.

---

## 7.9 What the Example Shows

The car example is simple, but it reveals the main logic of Actor-Critic methods.

### 1. The actor does not need a table of actions

The action is continuous acceleration. There is no need to enumerate every possible acceleration value. The policy directly represents a distribution over actions.

This is one reason policy-gradient and Actor-Critic methods are useful for control problems.

### 2. The critic turns experience into immediate feedback

After the first transition, the agent does not wait until the end of the episode. It immediately computes:

$$
\delta_0=-3.4.
$$

After the second transition, it immediately computes:

$$
\delta_1=5.335.
$$

Each transition gives learning signal to both actor and critic.

### 3. The actor learns relative quality, not raw reward

The second reward was still negative:

$$
r_1=-1.725.
$$

But the TD error was positive:

$$
\delta_1=5.335.
$$

This is important. Actor-Critic does not simply ask whether the immediate reward is positive. It asks whether the outcome was better than the critic expected.

A negative reward can still produce a positive update if it improves the future situation enough.

### 4. The critic can be wrong, especially early

At the beginning, the critic is only a rough approximation. Its estimates are based on initial weights, not on deep understanding. Therefore, the actor may initially learn from imperfect feedback.

This is the main cost of bootstrapping.

### 5. The example is linear, but the idea is general

We used:

$$
\mu_\theta(s)=\theta_1x+\theta_2v,
$$

and:

$$
V_w(s)=w_1x+w_2v.
$$

These are linear approximators. They are easy to calculate by hand. However, the same principle applies when the actor and critic are neural networks. The next chapter will explain why deep networks become necessary when states are too large or complex for hand-designed features.

---

## 7.10 Choosing the Critic’s Target

The critic is trained by comparing its current estimate to a target. In one-step Actor-Critic, the target is:

$$
y_t=r_t+\gamma V_w(s_{t+1}).
$$

The critic tries to make:

$$
V_w(s_t) \approx y_t.
$$

This target is called a **bootstrapped target** because it uses the critic’s own estimate of the next state.

This is different from supervised learning. In supervised learning, the target label is usually fixed. If we train an image classifier, the label “cat” or “dog” does not change because the network updated its weights.

In Actor-Critic and other TD methods, the target contains a learned estimate. When the critic changes, the target can also change. This makes reinforcement learning more unstable than ordinary supervised learning.

This idea becomes very important in Deep Reinforcement Learning. Once neural networks replace simple linear functions, changing one parameter can affect value estimates across many states at once.

---

## 7.11 Practical Design Choices

Even basic Actor-Critic requires several design choices. These choices matter because the actor and critic influence each other.

### 1. What should the critic estimate?

The critic can estimate state value:

$$
V(s),
$$

or action value:

$$
Q(s,a).
$$

A state-value critic is common when the policy-gradient update uses an advantage estimate. It asks: “How good is this state?”

An action-value critic asks: “How good is this state-action pair?”

The state-value critic is simpler for explaining the basic method. The action-value critic becomes more important in later continuous-control methods.

### 2. How fast should the actor and critic learn?

If the actor changes too quickly, the critic is always trying to evaluate a moving policy. If the critic changes too slowly, the actor receives outdated feedback.

A common intuition is:

- the critic should learn quickly enough to track the actor;
- the actor should learn cautiously enough not to chase noise.

This is sometimes called a two-timescale problem.

### 3. How much should the policy explore?

For stochastic policies, exploration is built into the action distribution. In the Gaussian example, $\sigma$ controls how widely actions are sampled.

If exploration is too small, the policy may converge too early to a poor behaviour. If exploration is too large, learning becomes noisy and unstable.

### 4. What happens during deployment?

During training, the actor needs the critic because the critic provides learning signals. During deployment, the agent may not need the critic to choose actions.

If the actor has learned a useful policy, we can act using:

$$
a\sim\pi_\theta(a\mid s),
$$

or use the mean action in a continuous policy:

$$
a=\mu_\theta(s).
$$

The critic is mainly a training tool. It helps shape the actor, but the actor is the component that directly controls behaviour.

---

## 7.12 Common Failure Modes

Actor-Critic methods are powerful, but they can fail in several predictable ways.

### 1. The critic gives misleading feedback

If $V_w(s)$ is inaccurate, then $\delta_t$ is inaccurate. The actor may reinforce actions that only appear good because the critic is wrong.

This is especially common early in training.

### 2. The actor changes the data distribution

The critic learns from states visited by the actor. But as the actor changes, the visited states also change. This means the critic’s training distribution is not fixed.

### 3. Bootstrapping can amplify errors

The target:

$$
r_t+\gamma V_w(s_{t+1})
$$

uses the critic’s own estimate. If the next-state estimate is too high, the current value may be pushed too high as well. Errors can propagate backward through the value function.

### 4. Linear approximators may be too weak

In the car example, the state was only:

$$
(x,v).
$$

A linear value function was enough for demonstration. But real states may be images, sensor histories, or high-dimensional continuous vectors. A linear function may not be expressive enough.

This limitation leads naturally to function approximation with neural networks.

---

## 7.13 The Remaining Limitation: Representation and Stability

The actor and critic in the worked example were deliberately written as simple linear functions:

$$
\mu_\theta(s)=\theta_1x+\theta_2v,\quad V_w(s)=w_1x+w_2v.
$$

This was only a teaching choice. It let us calculate the actor update, critic update, and TD error by hand. It should not be read as a limitation of Actor-Critic itself.

In general, the actor can be any differentiable policy approximator:

$$
\pi_\theta(a\mid s),
$$

and the critic can be any differentiable value approximator:

$$
V_w(s) \quad \text{or} \quad Q_w(s,a).
$$

Those approximators may be linear models, shallow neural networks, convolutional neural networks, recurrent networks, or other architectures. The Actor-Critic idea is the relationship between the two learners: the actor improves the policy, and the critic supplies a value-based learning signal.

The real limitation at this point is not that Actor-Critic must use simple functions. The limitation is that, in large environments, we need a representation powerful enough to generalize across states.

For a tiny state such as:

$$
s=(x,v),
$$

a hand-designed feature vector may be enough. But in realistic tasks, the state may be an image, a sensor stream, a long history of observations, or a high-dimensional vector. We do not want to manually design every useful feature. We want the agent to learn useful representations from data.

This is exactly the point where **[the next chapter on Deep Reinforcement Learning][deep-rl]** takes over. Deep learning does not “fix” Actor-Critic by replacing it; rather, it gives Actor-Critic and value-based methods more powerful function approximators. A deep actor can learn a policy from complex states. A deep critic can learn value estimates from complex states. The same actor-critic logic remains, but the representation becomes much richer.

However, deep function approximation also introduces a new problem. Bootstrapped targets such as:

$$
y_t=r_t+\gamma V_w(s_{t+1})
$$

are already moving targets, because they depend on a learned critic. When $V_w$ is a neural network, one parameter update can change predictions for many states at once. This can make training unstable.

So the chapter ends with a clear tradeoff:

- Actor-Critic improves over Monte Carlo methods by learning from shorter TD-style feedback.
- Actor-Critic improves over pure TD control by learning an explicit policy.
- But when states become large and complex, the actor and critic need stronger function approximators.
- Deep learning helps with this representation problem, while also creating new stability problems.

**[The next chapter][deep-rl]** focuses on that next step: how neural networks are used in reinforcement learning, why targets become unstable, and why techniques such as target networks and replay buffers become important.

---

## Summary and Key Takeaways

**Actor-Critic methods continue the policy-gradient story.** The previous chapter showed that a policy can be optimized directly. Actor-Critic keeps that idea but adds a critic to evaluate actions.

**The actor learns how to act.** It represents the policy $\pi_\theta(a\mid s)$ and is updated to make better-than-expected actions more likely.

**The critic learns how good states are.** It estimates $V_w(s)$ and provides a baseline for judging the actor’s actions.

**The TD error is the main learning signal.** It is defined as:

$$
\delta_t=r_t+\gamma V_w(s_{t+1})-V_w(s_t).
$$

It works like a one-step advantage estimate.

**Actor-Critic improves on ordinary Monte Carlo methods by learning online.** It does not need to wait for full episode returns, so it can learn from shorter feedback signals and can be used in continuing tasks.

**Actor-Critic improves on pure TD learning by learning an explicit policy.** TD learning gives the bootstrapped evaluation signal, while the actor gives direct policy optimization. This is especially useful when the policy should be stochastic or when the action space is continuous.

**Actor-Critic reduces variance but introduces bias.** It learns faster than pure Monte Carlo policy gradients because it uses immediate bootstrapped feedback. But if the critic is wrong, the actor learns from biased guidance.

**Continuous actions are natural in this framework.** A Gaussian policy can represent acceleration, torque, steering, or other real-valued controls without enumerating actions.

**The 1D car example shows the mechanics clearly.** A negative TD error reduces the probability of a sampled action. A positive TD error reinforces it, even if the immediate reward itself is still negative.

**The next step is Deep Reinforcement Learning.** The linear actor and critic in this chapter were used only to make the math transparent. Actor-Critic can also use neural networks. The remaining limitation is representation: large state spaces, images, and sensor streams require approximators that can learn useful features automatically. **[The next chapter][deep-rl]** addresses this by introducing deep neural-network function approximation, while also explaining the new stability problems that appear: moving targets, correlated samples, replay buffers, and target networks.


[finite-mdp]: ../chapter2/chapter2a.md
[q-learning]: ../chapter6/chapter6-part2.md
[td-learning]: ../chapter6/chapter6-part1.md
[policy-gradient]: ../chapter7/chapter7.md
[deep-rl]: ../chapter8/chapter8.md

---

## References

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, *SMC-13*(5), 834–846.

Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. *Advances in Neural Information Processing Systems*.

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, *3*, 9–44.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*.

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, *8*, 229–256.

---

## Citation

To cite this chapter, please use the following BibTeX:

```bibtex
@misc{saleh_2026_actor_critic_methods,
  author       = {Ahmed Said Saleh},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter 7 Part 2: Actor-Critic Methods},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}
```
