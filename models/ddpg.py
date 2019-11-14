import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from tqdm.auto import tqdm
from models import Agent


def ddpg(env: UnityEnvironment, agent=None, n_episodes=500, max_t=2000, eps_start=1.0,
         eps_end=0.01, eps_decay=0.995, seed=0, target=30.0):
    """
    Deep Deterministic Policy Gradients
    
    Params
    ======
        env (UnityEnvironment): enviroment
        agent (Agent): agent that is responsible for control the actions (if no agent, create a new agent)
        n_episodes (int): maximum number of training episodes
        max_t (int): max number of steps in each episode
        eps_start (float): starting value of epsilon, for attenuate the noise applied to the action
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        seed (int): random seed
        target (float): value expected to save the model and exit training

    Return
    ======
        scores: List all scores (for all the agents) in each episode
        agent (Agent): Trained agent
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get size of action and state
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    if not agent:                                   # create the agent if no agent is provided
        agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)
    agent.network_summary()                         # Print Networks Info (actor and critic)

    all_scores = []                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)               # last 100 scores
    eps = eps_start                                 # initialize epsilon

    print('*** Starting Training ***')
    for i_episode in tqdm(range(1, n_episodes + 1)):
        scores = run_single_episode(env, brain_name, agent, train_mode=True, epsilon=eps, max_t=max_t)
        scores_window.append(scores)                # save most recent score to the window
        all_scores.append(scores)                   # save most recent score
        eps = max(eps_end, eps_decay * eps)         # decrease epsilon

        print(score_log(i_episode, scores_window, scores), end="")
        # Save the model each 100 or if the target is achieved
        if i_episode % 100 == 0 or np.mean(scores_window) >= target:
            print(score_log(i_episode, scores_window, scores) + '\tSaved!')
            agent.save()
            if np.mean(scores_window) >= 30:
                break                               # Finish the execution if the target is achieved

    return all_scores, agent


def run_single_episode(env: UnityEnvironment, brain_name, agent: Agent=None, train_mode=False, max_t=2000, epsilon=0.0):
    """
    Execute a single episode

    Params
    ======
        env (UnityEnvironment): enviroment
        brain_name (string): default brain name
        agent (Agent): agent that is responsible for control the actions (if no agent, a random action is chosen)
        train_mode (bool): indicate if the environment is on the train mode
        max_t (int): max number of steps in each episode
        epsilon (float): attenuate the noise applied to the action

    Return
    ======
        scores (float): episode scores of all agents
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_agents = len(env_info.agents)
    action_size = env.brains[brain_name].vector_action_space_size

    states = env_info.vector_observations
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    # Run all the steps of one episode
    for time_step in range(1, max_t+1):
        if agent:                                       # if a agent is provide, get all the action (for each agent)
            actions = agent.act(states, epsilon=epsilon, add_noise=train_mode)
        else:                                           # select a random action (if no agent)
            actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)               # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]        # send the action to the environment

        next_states = env_info.vector_observations      # get the next state
        rewards = env_info.rewards                      # get the reward
        dones = env_info.local_done                     # get the done flag

        if agent and train_mode:                        # learn if in train mode
            agent.step(states, actions, rewards, next_states, dones, time_step)
            
        states = next_states
        scores += rewards                               # increase the scores (for each agent)
        if np.any(dones):                               # Exit episode if done
            break

    return scores


def score_log(i, s, e):
    """Log score"""
    return f'\rEpisode {i}\tAverage Score: {np.mean(s):.2f}\tLast Episode Score: {np.mean(e):.2f}'
