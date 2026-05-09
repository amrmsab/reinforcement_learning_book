# How RL recommenders work

Most surveys explain this RL recommendation interaction through the framework of a Markov Decision Process, or MDP. An MDP is a way to describe decision-making step by step. In recommender systems, the recommender can be viewed as the agent, while the user and the surrounding platform form the environment.

![RL Model](<Images/RL slide 3.png>)

## State

The first part of the formulation is the state. In plain terms, the state is what the system currently knows about the user and the situation. This may include the user’s recent clicks, watch history, purchases, search behavior, session context, or learned embeddings that summarize past behavior.

## Action

The second part is the action. In recommendation, an action usually means choosing an item to show the user. On Amazon, the action could be recommending a specific product. On YouTube, it may be selecting the next video to place on the home page.

## Reward

The third part is the reward. This is the signal that tells the system whether its recommendation was helpful or not.

## State Transition

The fourth part is the transition from one state to another. After the system recommends something, the user reacts, and that reaction changes the situation. If a user clicks a product, adds it to a cart, ignores it, or leaves the app, the next recommendation should be different. This changing state is what makes recommendation dynamic.