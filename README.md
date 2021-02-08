# Breakout-v4
## Description
Breakout-v4 is an OpenAI gym environment in which the goal is to maximize your score in the Atari 2600 game Breakout. In this environment, the observation is an RGB image of the screen and each action is repeatedly performed for a duration of *k* frames, where *k* is uniformly sampled from *{2,3,4}*. There is 0 probability of repeating the previous action (unlike Breakout-v0 which has a probability of 0.25). I have used a variant of it called **BreakoutNoFrameskip-v4** which has no frame skip and no action repeat stochasticity as we introduce both manually.  
   
I have implemented a Deep Q-Network agent with the help of the *tf-agents* library.

## Prerequisites & Usage
Pyhton 3 is required. The libraries and their versions required can be found in the corresponding [requirements.txt](requirements.txt) file. They can be installed with the following command in the terminal:   
`pip3 install -r requirements.txt`   

As can be seen, there are two Python files
- [agent.py](agent.py): contains the agent class and all the necessary functions. 
- [train.py](train.py): used for creation, training and evaluation of the agent.   
   
After the requirements are satisifed, run the the Python file in terminal with:    
`python3 train.py` 

## Observation
## Result

