# RL Recommender Methods

Now that recommendation is framed as a **sequential decision problem**, the next question is: *which reinforcement learning method should be used?*  

Broadly, RL approaches for recommender systems fall into two main categories:

- **Model-Free methods**
- **Model-Based methods**

## Model-Free Methods

Model-free reinforcement learning methods learn directly from interaction data without explicitly modeling how the environment evolves after each action.  

In recommender systems, this means the algorithm improves its recommendations based on observed user feedback such as:

- Clicks  
- Purchases  
- Skips  
- Watch time  

rather than first building a model of user behavior.

Two well-known examples are:

- **Q-learning**
- **SARSA**

![Q-learning vs SARSA](Images/sarsa%20and%20q%20learning.png)

### Limitations

Despite their simplicity, model-free methods face significant challenges in large-scale recommender systems:

- The number of users and items can be extremely large  
- The action space becomes too complex for simple methods  
- Learning can become inefficient and slow  

As a result, naive model-free approaches often struggle to scale in real-world applications.

## Model-Based Methods

Model-based reinforcement learning takes a different approach. Instead of relying solely on trial-and-error learning, these methods first build a model of user behavior.

This model can capture:

- How users respond to recommendations  
- How interaction sequences evolve over time  

Using this model, the system can **simulate outcomes** and plan better recommendations before taking action.

### Advantages

- More efficient learning (fewer real interactions required)  
- Ability to plan ahead using predicted outcomes  
- Better handling of long-term effects  

### Challenges

- Building accurate user behavior models is difficult  
- Systems become more complex to design and maintain  

Because of this, model-based RL is **theoretically appealing** but often **harder to apply in practice**.