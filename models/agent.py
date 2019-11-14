import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment
from models.actor import Actor
from models.critic import Critic
from models.helper import path_result_folder
from models.parameters import device, LR_ACTOR, LR_CRITIC, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, WEIGHT_DECAY, \
    LEARN_STEPS, N_UPDATES
from models.replay_buffer import ReplayBuffer


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        np.random.seed(random_seed)                         # set the numpy seed

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)

    def step(self, states, actions, rewards, next_states, dones, time_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward (for each agent)
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory and every 20 steps
        if len(self.memory) > BATCH_SIZE and time_step % LEARN_STEPS == 0:
            for _ in range(N_UPDATES):          # generate n experiences and realize n updates
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, epsilon=0.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:                           # add a noise (based on normal distribution) to exploration
            actions += np.random.normal(0, .3) * epsilon

        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        self.__update_critic_local(actions, dones, gamma, next_states, rewards, states)
        self.__update_actor_local(states)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def __update_critic_local(self, actions, dones, gamma, next_states, rewards, states):
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
    def __update_actor_local(self, states):
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def network_summary(self):
        print('- Actor Summary (both local and target): ')
        self.actor_local.to(device).summary()

        print('- Critic Summary (both local and target): ')
        self.actor_local.to(device).summary()

    def save(self, checkpoint_actor_name='checkpoint_actor', checkpoint_critic_name='checkpoint_critic'):
        """Save the actor and critic network weights"""
        torch.save(self.actor_local.state_dict(), path_result_folder(f'{checkpoint_actor_name}.pth'))
        torch.save(self.critic_local.state_dict(), path_result_folder(f'{checkpoint_critic_name}.pth'))

    @staticmethod
    def load(env: UnityEnvironment, random_seed=0, checkpoint_actor_name='checkpoint_actor', checkpoint_critic_name='checkpoint_critic'):
        """Load the actor and critic network weights"""
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        env_info = env.reset(train_mode=True)[brain_name]
        state_size = len(env_info.vector_observations[0])
        action_size = brain.vector_action_space_size

        loaded_agent = Agent(state_size, action_size, random_seed)
        loaded_agent.actor_local.load_state_dict(torch.load(path_result_folder(f'{checkpoint_actor_name}.pth')))
        loaded_agent.critic_local.load_state_dict(torch.load(path_result_folder(f'{checkpoint_critic_name}.pth')))
        return loaded_agent
