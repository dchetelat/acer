import torch
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

        trajectory = self.explore(actor_critic) if on_policy else self.buffer.sample()

        _, _, _, next_states, done, _, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(next_states)))
        retrace_action_value = (action_probabilities * action_values).data.sum() * (1. - done)

        for states, actions, rewards, _, done, _, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(states)))
            value = (action_probabilities * action_values).data.sum() * (1. - done)
            importance_weights = action_probabilities.data / exploration_probabilities

            naive_advantage = action_values.data[actions] - value

            retrace_action_value = rewards + DISCOUNT_FACTOR * retrace_action_value
            retrace_advantage = retrace_action_value - value

            actor_loss = - min(TRUNCATION_PARAMETER, importance_weights[actions]) * \
                          retrace_advantage * action_probabilities[actions].log()
            actor_loss += - (Variable((1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
                                      naive_advantage * action_probabilities.data) * action_probabilities.log()).sum()

            critic_loss = (action_values[actions] - retrace_action_value).pow(2)

            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)

            retrace_action_value = min(1., importance_weights[actions]) * \
                                   (retrace_action_value - action_values.data[actions]) + value
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()

    def explore(self, actor_critic, max_steps=100):
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
        trajectory = []
        rewards = 0.
        for step in range(max_steps):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(state)))

            action = action_probabilities.multinomial()
            action = action.data.numpy()[0]
            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            transition = replay_memory.Transition(state, action, reward, next_state,
                                                  float(done), 0., action_probabilities.data)
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
