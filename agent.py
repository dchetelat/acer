import torch
import numpy as np
import replay_memory
from torch.autograd import Variable
from brain import ActorCritic
from core import *


class Agent:
    """
    Agent that learns an optimal policy using the ACER reinforcement learning algorithm.

    Parameters
    ----------
    """
    def __init__(self, brain, render=False):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.render = render
        self.buffer = replay_memory.ReplayBuffer()
        self.brain = brain
        self.optimizer = torch.optim.Adam(brain.actor_critic.parameters())

    def learn(self, on_policy):
        actor_critic = ActorCritic()
        actor_critic.copy_parameters_from(self.brain.actor_critic)

        trajectory = self.explore(actor_critic, MAX_STEPS_BEFORE_UPDATE) \
                     if on_policy else self.buffer.sample(OFF_POLICY_MINIBATCH_SIZE)

        _, _, _, next_states, done, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(next_states)))
        retrace_action_value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(states)))
            value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1) * torch.from_numpy(1. - done)
            importance_weights = action_probabilities.data / torch.from_numpy(exploration_probabilities)
            action_indices = Variable(torch.LongTensor(actions))

            naive_advantage = action_values.gather(-1, action_indices).data - value

            retrace_action_value = torch.FloatTensor(rewards) \
                                   + DISCOUNT_FACTOR * retrace_action_value * torch.from_numpy(1. - done)
            retrace_advantage = retrace_action_value - value

            actor_loss = - Variable(importance_weights.gather(-1, action_indices.data).clamp(max=TRUNCATION_PARAMETER)
                           * retrace_advantage) * action_probabilities.gather(-1, action_indices).log()
            actor_loss += - (Variable((1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
                                      naive_advantage * action_probabilities.data) * action_probabilities.log()).sum()

            critic_loss = (action_values.gather(-1, action_indices) - Variable(retrace_action_value)).pow(2)

            actor_loss.mean().backward(retain_graph=True)
            critic_loss.mean().backward(retain_graph=True)

            retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) * \
                                   (retrace_action_value - action_values.gather(-1, action_indices).data) + value
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()

    def explore(self, actor_critic, max_steps):
        """
        Explore an environment by taking a sequence of actions and saving the results in the memory.

        Parameters
        ----------
        actor_critic : ActorCritic
            The actor-critic model to use to explore.
        max_steps : int
            The maximum number of steps to take at a time.
        """
        state = self.env.env.state
        if isinstance(state, tuple):
            state = np.array(list(state))
        trajectory = []
        rewards = 0.
        for step in range(max_steps):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(state)))

            action = action_probabilities.multinomial()
            action = action.data.numpy()
            next_state, reward, done, _ = self.env.step(action[0])
            if not isinstance(next_state, np.ndarray):
                raise ValueError
            if self.render:
                self.env.render()
            transition = replay_memory.Transition(states=state.reshape(1, -1),
                                                  actions=action.reshape(1, -1),
                                                  rewards=np.array([[reward]], dtype=np.float32),
                                                  next_states=next_state.reshape(1, -1),
                                                  done=np.array([[done]], dtype=np.float32),
                                                  action_probabilities=action_probabilities.data.numpy().reshape(1, -1))
            self.buffer.add(transition)
            trajectory.append(transition)
            rewards += reward
            if done:
                self.env.reset()
                break
            else:
                state = next_state
        if self.render:
            print(", total rewards {}".format(rewards))
        return trajectory
