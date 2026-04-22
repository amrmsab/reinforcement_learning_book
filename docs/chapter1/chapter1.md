# Introduction to Reinforcement Learning

## What is Reinforcement Learning
 Reinforcement Learning (RL) is a branch of Machine Learning (ML) that focuses on how an agent should take actions in an environment to maximize the reward it receives.

 RL refers to a type of problem, a class of solution methods that work well on these problems, and a field of study that focuses on these problems and solution methods.

### Distinguishing features of an RL problem
1. RL problems are closed-loop systems. A closed-loop system means the agent's actions influence its later inputs.
2. There is no supervisor that guides the agent on which action to take.
3. The consequences of actions play out over extended time periods.

<div align="center">
  <img src="Chapter%201%20Closed%20loop%20figure%20sutton%20and%20barto.webp" alt="Closed Loop" width="60%">
  <p style="font-size: 0.9em;"><em>Figure 1.1: An agent-environment closed-loop diagram</em></p>
</div>

## Formulation of an RL problem
An RL agent requires an explicit goal dependent on its environment's state. Therefore, the agent must sense the state of the environment (to some extent) and possess the ability to affect the state through actions. These three aspects — sensation, action, and a goal — are the simplest form of an RL problem. Any method that is well-suited to solve problems with these aspects can be considered an RL solution method.

## Comparison with other learning paradigms

### Supervised Learning

<div align="center">
  <img src="supervised-machine-learning.webp" alt="Supervised Learning" width="60%">
  <p style="font-size: 0.9em;"><em>Figure 1.2: Supervised Learning</em></p>
</div>

Supervised learning is learning from a training set of labeled examples provided by a knowledgeable external supervisor. Each example is a description of a situation or state of the environment, and its label is the correct action the agent must do in that situation. It is often used to classify the input into the category to which it belongs. The objective of supervised learning is for the agent or system to generalize its responses so that it acts correctly in situations not present in the training set. 



This type of learning is not adequate for learning from interaction. In interactive problems, it is often impractical to obtain examples of desired behavior that are both correct and representative of all situations in which the agent has to act. In uncharted territory, we would like the agent to learn from its own experience.

### Unsupervised Learning

<div align="center">
  <img src="Unsupervised-Learning.jpg" alt="Unsupervised Learning" width="60%">
  <p style="font-size: 0.9em;"><em>Figure 1.3: Unsupervised Learning</em></p>
</div>

Unsupervised learning is typically about finding structure hidden in collections of unlabeled data. Even though one might be tempted to think of RL as an unsupervised learning method, they differ in their goals; unsupervised learning tries to find hidden structure, while RL does not. We, therefore, consider RL to be a third ML paradigm.

### Unique challenges in RL
There are challenges that arise in RL and not in other types of learning, such as the trade-off between exploration and exploitation. For the agent to obtain a high reward, it must prefer actions that it has tried in the past and found to be effective in producing reward. However, to discover such actions, it has to try actions that it has not selected before. The agent has to exploit what it already knows in order to obtain reward, but also has to explore in order to make better action selections in the future. The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing the task. The agent must try a variety of actions and progressively favor those that appear to be best.

Another key feature of RL is that it explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment. This is different from many approaches which consider subproblems without addressing how they might fit into the larger picture, such as classifiers trained with supervised learning.

### Examples of RL problems

<div align="center">
  <img src="gazelle.avif" alt="Gazelle Calf learning to run" width="60%">
  <p style="font-size: 0.9em;"><em>Figure 1.4: Gazelle calf learning to run</em></p>
</div>

RL starts with a complete interactive goal-seeking agent. The agent has an explicit goal, can sense its environment, and can choose actions that affect the state of the environment. It is usually assumed that the agent has to operate despite uncertainty about the environment. For example, when a chess player makes a move, their choice is informed both by planning and by intuitive judgment of the desirability of the new position.

A gazelle calf struggles to its feet minutes after being born, yet half an hour later it is running at two miles per hour. This example show the agent interacting with its environment to achieve a goal, despite uncertainty. Correct choices require taking into account the indirect, delayed consequences of actions and thus may require foresight or planning. Also, the effects of actions cannot be fully predicted; thus, the agent must monitor its environment frequently and react appropriately.

## Main Subelements of an RL system
There are four main subelements that make up an RL system: a policy, a reward signal, a value function, and optionally a model of the environment. The model is optional because not all RL methods require planning.

### 1. Policy
A policy defines the agent's way of behaving at a given time. It is a mapping from perceived states to actions to be taken in these states, corresponding to a set of stimulus-response rules. The policy may be a simple function or a lookup table, or it might involve extensive computation such as a search process.

The policy is the core of an RL agent in the sense that it alone is sufficient to determine its behavior.

### 2. Reward Signal
A reward signal defines the goal in an RL problem. On each time step, the environment sends to the agent a single number, the reward. The agent's sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. The reward sent to the agent depends on the agent's action and the state of the environment. The agent cannot directly alter this process, so the only way the agent can influence the reward signal is through its actions, which can have a direct effect on the reward or an indirect effect through changing the environment state. The reward signal is the primary basis for altering the policy.

### 3. Value Function
The value function specifies what is good in the long run. The value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. It indicates the long-term desirability of states after taking into account the rewards available in those states.

Rewards are in a sense primary, whereas values—which are predictions and aggregations of future rewards—are secondary. Without rewards, there could be no value, and the only purpose of values is to accumulate more reward. Nevertheless, it is values that we are most concerned about when making and evaluating decisions. Action choices are made based on value judgments because these metrics obtain the greatest amount of reward over the long run. Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are given directly by the environment, whereas values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime. In fact, the most important component of all RL algorithms is a method for effectively estimating values.

### 4. Model of the Environment
The model is something that mimics the behavior of the environment, allowing inferences to be made about how the environment will behave. Models are used for planning. RL problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-and-error learners.


With these four subelements in place, we can now begin to formalize the RL framework mathematically — which we will do in the next chapter.

## Citations
* Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
* Hess, Shervin. "Speke's Gazelle Juliet Runs in the Africa Savanna Habitat." The Oregonian/OregonLive, Oregon Zoo, <https://www.oregonlive.com/living/2016/04/baby_gazelle_that_nearly_died.html>
* GeeksforGeeks. "Supervised Machine Learning" GeeksforGeeks, 14 Apr. 2026, <https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/>.
* WisdomPlexus. "Supervised Learning vs Unsupervised Learning: Key Differences To Know." WisdomPlexus, 31 May 2024, <https://www.wisdomplexus.com>.
