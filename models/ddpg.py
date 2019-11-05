import torch
import os
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from tqdm.auto import tqdm

from models import Agent


def ddpg(env: UnityEnvironment, n_episodes=200, max_t=2000, eps_start=1.0,
         eps_end=0.01, eps_decay=0.995, seed=0, save_threshold=0.0):
    """
    Deep Deterministic Policy Gradients
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        seed (int): random seed
        save_threshold (float): value expected to save the model and exit training
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get size of action and state
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    # create the agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    all_scores = []                         # list containing scores from each episode
    scores_window = deque(maxlen=100)       # last 100 scores
    eps = eps_start                         # initialize epsilon
    for i_episode in tqdm(range(1, n_episodes+1)):
        scores = run_single_episode(env, brain_name, agent, max_t, train_mode=True)
        scores_window.append(scores)        # save most recent score
        all_scores.append(scores)           # save most recent score
        eps = max(eps_end, eps_decay*eps)   # decrease epsilon

        print(score_log(i_episode, scores_window, scores), end="")
        if i_episode % 100 == 0:
            if np.mean(scores_window) >= save_threshold:    # save if avg score is higher than the threshold
                print(score_log(i_episode, scores_window, scores) + '\tSaved!')
                agent.save()
                # break
            else:
                print(score_log(i_episode, scores_window, scores))

    return all_scores, agent


def run_single_episode(env: UnityEnvironment, brain_name, agent: Agent=None, max_t=2000, train_mode=False):
    """
    Execute a single episode

    Params
    ======
        env (UnityEnvironment): enviroment
        brain_name (string): default brain name
        agent (Agent): agent that is responsible for control the actions (if no agent, a random action is chosen)
        max_t (int): max steps in each episode
        train_mode (bool): indicate if the environment is on the train mode

    Return
    ======
        score (float): total score of episode
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_agents = len(env_info.agents)
    action_size = env.brains[brain_name].vector_action_space_size

    states = env_info.vector_observations
    if agent:
        agent.reset()
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    for time_step in range(max_t):  # Run each step in episode
        if agent:
            actions = agent.act(states)
        else:
            actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]        # send the action to the environment

        next_states = env_info.vector_observations   # get the next state
        rewards = env_info.rewards                   # get the reward
        dones = env_info.local_done                  # get the done flag

        if agent and train_mode:
            agent.step(states, actions, rewards, next_states, dones, time_step)
            
        states = next_states
        scores += rewards
        if np.any(dones):    # Exit episode if done
            break

    return scores


def score_log(i, s, e):
    """Log score"""
    return f'\rEpisode {i}\tAverage Score: {np.mean(s):.2f}\tLast Episode Score: {np.mean(e):.2f}'
