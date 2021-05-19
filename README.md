[image1]: assets/eq_1.png "image1"
[image2]: assets/eq_2.png "image2"
[image3]: assets/eq_3.png "image3"
[image4]: assets/eq_4.png "image4"
[image5]: assets/eq_5.png "image5"
[image6]: assets/eq_6.png "image6"



# Deep Reinforcement Learning Theory - Policy Based Methods

## Content 
- [Introduction](#intro)
- [What are Policy Gradient Methods?](#What_are_Policy_Gradient_Methods)
- [The Big Picture](#The_Big_Picture)
- [Problem Setup](#Problem_Setup)
- [REINFORCE](#REINFORCE)
- [Derivation](#Derivation)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
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

## What are Policy Gradient Methods? <a name="What_are_Policy_Gradient_Methods"></a> 
- ***Policy-based methods** are a class of algorithms that search directly for the optimal policy, without simultaneously maintaining value function estimates.
- ***Policy gradient methods*** are a subclass of policy-based methods that estimate the weights of an optimal policy through ***gradient ascent***.
- Here, we use the policy with a neural network, where the goal to find the weights **θ** of the network that maximize expected return.

## The Big Picture <a name="The_Big_Picture"></a>
- The policy gradient method will iteratively amend the policy network weights to:
    - make (state, action) pairs that resulted in positive return more likely, and
    - make (state, action) pairs that resulted in negative return less likely.

## Problem Setup <a name="Problem_Setup"></a>
- A **trajectory** **τ** is a state-action sequence **s<sub>0</sub>, a<sub>0</sub>, …, s<sub>H</sub>, a<sub>H</sub>, s<sub>H+1</sub>**.
- Use the notation **R(τ)** to refer to the return corresponding to trajectory **τ**.
- Goal is to find the weights **θ** of the policy network to maximize the expected return **U(θ) := ∑<sub>τ</sub> P(τ;θ) R(τ)**.

## REINFORCE <a name="REINFORCE"></a>
- The pseudocode for REINFORCE is as follows:
    1. Use the policy **π<sub>θ</sub>** to collect **m** trajectories **{τ<sup>(1)</sup>, τ<sup>(2)</sup>, …, τ<sup>(m)</sup>}** with horizon **H**. We refer to the i-th trajectory as 
    
        ![image1]
    
    2. Use the trajectories to estimate the gradient **∇<sub>θ</sub>U(θ)**
    
        ![image2]

    3. Update the weights of the policy:

        ![image3]

    4. Loop over step 1-3

## Derivation <a name="Derivation"></a>
- We derived the likelihood ratio policy gradient:

    ![image6]

- We can approximate the gradient above with a sample-weighted average: 

    ![image4]

- We calculated the following: 

    ![image5]

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
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

### Clone the project <a name="Clone_the_project"></a>
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

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

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
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
