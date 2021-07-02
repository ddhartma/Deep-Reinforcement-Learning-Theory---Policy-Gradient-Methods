[image1]: assets/eq_1.png "image1"
[image2]: assets/eq_2.png "image2"
[image3]: assets/eq_3.png "image3"
[image4]: assets/eq_4.png "image4"
[image5]: assets/eq_5.png "image5"
[image6]: assets/eq_6.png "image6"
[image7]: assets/introduction.png "image7"
[image8]: assets/big_picture.png "image8"
[image9]: assets/rl_sl.png "image9"
[image10]: assets/problem_setup.png "image10"
[image11]: assets/expectation.png "image11"
[image12]: assets/update.png "image12"
[image13]: assets/reinforce.png "image13"
[image14]: assets/code_result.png "image14"
[image15]: assets/cart_pole.png "image15"

# Deep Reinforcement Learning Theory - Policy Gradient Methods

## Content
- [Introduction](#intro)
- [What are Policy Gradient Methods?](#What_are_Policy_Gradient_Methods)
- [The Big Picture](#The_Big_Picture)
- [Problem Setup](#Problem_Setup)
- [REINFORCE](#REINFORCE)
- [Derivation](#Derivation)
- [REINFORCE - Code Implementation](#reinforce_code)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a id="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems

## What are Policy Gradient Methods? <a id="What_are_Policy_Gradient_Methods"></a>
- ***Policy-based methods*** are a class of algorithms that search directly for the optimal policy, without simultaneously maintaining value function estimates.
- ***Policy gradient methods*** are a subclass of policy-based methods that estimate the weights of an optimal policy through ***gradient ascent***.
- Here, we use the policy with a neural network, where the goal to find the weights **θ** of the network that maximize expected return.

    ![image7]

## The Big Picture <a id="The_Big_Picture"></a>
- The policy gradient method will iteratively amend the policy network weights to:
    - make (state, action) pairs that resulted in positive return more likely, and
    - make (state, action) pairs that resulted in negative return less likely.

    ![image8]

### Reinforcement LEarning vs. Supervised Learning:
- For each state action pair in the episode we add Gaussian noise to the weights to make it slightly more to select this action when the agent is in this state
- This is comparable to Supervised Learning:
    - If the networks makes a wrong prediction, weights are slightly changed to make better prediction in the next trial.
    - In both cases we have Input/Output pairs (RL: state-action pairs, SL: image-label pairs)
    - In RL are conflicting situations about choosing best action for given state. In SL: Same image appears twice in the dataset (one as dog and one as cat label)
- Differences:
    - In image classification: Data set does not change over time
    - In RL: The Data set varies by episode. We use the policy to collect an episode which gives us a datset (bunch of matched state action pairs). We use that data set once to do a batch of updates

    ![image9]

- Connections between policy gradient methods and supervised learning, check out [Andrej Karpathy's blog post](http://karpathy.github.io/2016/05/31/rl/)

## Problem Setup <a id="Problem_Setup"></a>
- A **trajectory** **τ** is a state-action sequence (without rewards) **s<sub>0</sub>, a<sub>0</sub>, …, s<sub>H</sub>, a<sub>H</sub>, s<sub>H+1</sub>**.
- No length restrictions: epside or small part of it.
- Length: H (=Horizon)
- Use the notation **R(τ)** to refer to the return corresponding to trajectory **τ**.

    ![image10]

- Goal is to find the weights **θ** of the policy network to **maximize the expected return** ([expectation](https://en.wikipedia.org/wiki/Expected_value))

    ![image11]

- Return **R(τ)** is a function of the trajectory **τ**. Then, we calculate the weighted average (where the weights are given by **P(τ;θ)** of all possible values that the return **R(τ)** can take.


## REINFORCE <a id="REINFORCE"></a>
- **Goal** is to find the values of the weights **θ** in the neural network that maximize the expected return **U**, where **τ** is an arbitrary trajectory. One way to determine the value of **θ** that maximizes this function is through **gradient ascent**.
- This algorithm is closely related to gradient descent, where the differences are that:

    - gradient descent is designed to find the **minimum** of a function, whereas gradient ascent will find the **maximum**, and
    - **gradient descent** steps in the direction of the **negative gradient**, whereas gradient ascent steps in the **direction of the gradient**.
- Update rule:

    ![image12]

    where **α** is the step size that is generally allowed to decay over time. Once we know how to calculate or estimate this gradient, we can repeatedly apply this update step, in the hopes that **θ** converges to the value that maximizes **U(θ)**

    ![image13]

- The pseudocode for REINFORCE is as follows:
    1. Use the policy **π<sub>θ</sub>** to collect **m** trajectories **{τ<sup>(1)</sup>, τ<sup>(2)</sup>, …, τ<sup>(m)</sup>}** with horizon **H**. We refer to the i-th trajectory as

        ![image1]

    2. Use the trajectories to estimate the gradient **∇<sub>θ</sub>U(θ)**

        ![image2]

    3. Update the weights of the policy:

        ![image3]

    4. Loop over step 1-3



## Derivation <a id="Derivation"></a>
- We derived the likelihood ratio policy gradient:

    ![image6]

- We can approximate the gradient above with a sample-weighted average:

    ![image4]

- We calculated the following:

    ![image5]

## REINFORCE - Code Implementation <a id="reinforce_code"></a>
- Open Jupyter Notebook ```reinforce.ipynb```
    ### Import the Necessary Packages
    ```
    import gym
    gym.logger.set_level(40) # suppress warnings (please remove if gives error)
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    import torch
    torch.manual_seed(0) # set random seed
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical

    !python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    ```
    ### Define the Architecture of the Policy
    ```
    env = gym.make('CartPole-v0')
    env.seed(0)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Policy(nn.Module):
        """ Define a Policy
        """
        def __init__(self, s_size=4, h_size=16, a_size=2):
            """ Initialize weights in the policy arbitrarily

                INPUTS:
                ------------
                    s_size - (int) size of state space
                    h_size - (int) number of hidden units
                    a_size - (int) size of action space

                OUTPUTS:
                ------------
                    no direct

            """
            super(Policy, self).__init__()
            self.fc1 = nn.Linear(s_size, h_size)
            self.fc2 = nn.Linear(h_size, a_size)

        def forward(self, x):
            """ Create forward pass of neural network.

                INPUTS:
                ------------
                    x - (torch tensor)

                OUTPUTS:
                ------------
                    output - (torch tensor) softmax classification output
            """
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            output = F.softmax(x, dim=1)
            return output

        def act(self, state):
            """ Execute forward pass get best action for given state

                INPUTS:
                ------------
                    state - (numpy array) of state values (4 values)

                OUTPUTS:
                ------------
                    action_item - (int) 0 or 1
                    log_prob - (torch tesnsor) returns log of the probability for an action, e.g. tensor([-0.6009]),
                                            minus needed because of gradient ascent
                                            --> loss = -m.log_prob(action) * reward
            """
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            probs = self.forward(state).cpu()
            m = Categorical(probs)
            action = m.sample()
            action_item = action.item()
            log_prob = m.log_prob(action)
            return action_item, log_prob

    RESULTS:
    -------------
    observation space: Box(4,)
    action space: Discrete(2)
    ```
    ### Train the Agent with REINFORCE
    ```
    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
        """ Implementation of REINFORCE algorithm

            INPUTS:
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                gamma - (float) discount rate
                print_every - (int) how often to print average score (over last 100 episodes)

            OUTPUTS:
            ------------
                scores - (list) of accumulated rewards
        """
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = env.reset()
            for t in range(max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])

            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                break

        return scores

    scores = reinforce()
    ```
    ![image14]

    ### Watch a Smart Agent!
    ```
    env = gym.make('CartPole-v0')

    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for t in range(1000):
        action, _ = policy.act(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()
    ```
    ![image15]


## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a id="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [An Introduction to Deep Reinforcement Learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)
* Helpful medium blog post on policies [Off-policy vs On-Policy vs Offline Reinforcement Learning Demystified!](https://kowshikchilamkurthy.medium.com/off-policy-vs-on-policy-vs-offline-reinforcement-learning-demystified-f7f87e275b48)
* [Understanding Baseline Techniques for REINFORCE](https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Cheat Sheet](https://towardsdatascience.com/reinforcement-learning-cheat-sheet-2f9453df7651)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

Important publications
* [2004 Y. Ng et al., Autonomoushelicopterflightviareinforcementlearning --> Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~jordan/papers/ng-etal03.pdf)
* [2004 Kohl et al., Policy Gradient Reinforcement Learning for FastQuadrupedal Locomotion --> Policy Gradient Methods](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
* [2013-2015, Mnih et al. Human-level control through deep reinforcementlearning --> DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2014, Silver et al., Deterministic Policy Gradient Algorithms --> DPG](http://proceedings.mlr.press/v32/silver14.html)
* [2015, Lillicrap et al., Continuous control with deep reinforcement learning --> DDPG](https://arxiv.org/abs/1509.02971)
* [2015, Schulman et al, High-Dimensional Continuous Control Using Generalized Advantage Estimation --> GAE](https://arxiv.org/abs/1506.02438)
* [2016, Schulman et al., Benchmarking Deep Reinforcement Learning for Continuous Control --> TRPO and GAE](https://arxiv.org/abs/1604.06778)
* [2017, PPO](https://openai.com/blog/openai-baselines-ppo/)
* [2018, Bart-Maron et al., Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb)
* [2013, Sergey et al., Guided Policy Search --> GPS](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
* [2015, van Hasselt et al., Deep Reinforcement Learning with Double Q-learning --> DDQN](https://arxiv.org/abs/1509.06461)
* [1993, Truhn et al., Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
* [2015, Schaul et al., Prioritized Experience Replay --> PER](https://arxiv.org/abs/1511.05952)
* [2015, Wang et al., Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [2016, Silver et al., Mastering the game of Go with deep neural networks and tree search](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
* [2017, Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
* [2016, Mnih et al., Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [2017, Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [2017, Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [2016, Wang et al., Sample Efficient Actor-Critic with Experience Replay --> ACER](https://arxiv.org/abs/1611.01224)
* [2017, Lowe et al. Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
* [2017, Silver et al. Mastering the Game of Go without Human Knowledge --> AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
* [2017, Silver et al., Mastering Chess and Shogi by Self-Play with aGeneral Reinforcement Learning Algorithm --> AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)
