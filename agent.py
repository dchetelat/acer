import torch
import numpy as np
import replay_memory
from torch.autograd import Variable
from brain import ActorCritic
from core import *


class Agent:
    """
    Agent that learns an optimal policy using ACER.

    Parameters
    ----------
    brain : brain.Brain
        The brain to update.
    render : boolean, optional
        Should the agent render its actions in the on-policy phase?
    """
    def __init__(self, brain, render=False):
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.render = render
        self.buffer = replay_memory.ReplayBuffer()
        self.brain = brain
        self.optimizer = torch.optim.Adam(brain.actor_critic.parameters())

    def run_episode(self):
        """
        Run a complete episode. Analogue of Algorithm 1 in the paper.
        """
        episode_rewards = 0.
        end_of_episode = False
        while not end_of_episode:
            trajectory_rewards, end_of_episode = self.learning_iteration(on_policy=True)
            episode_rewards += trajectory_rewards
            for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
                self.learning_iteration(on_policy=False)
        if self.render:
            print(", episode rewards {}".format(episode_rewards))

    def learning_iteration(self, on_policy):
        """
        Conduct a single learning iteration. Analogue of Algorithm 2 in the paper.

        Parameters
        ----------
        on_policy : boolean
            Should the iteration be on-policy or off-policy?

        Returns
        -------
        trajectory_rewards : float or None
            The total rewards accumulated during the iteration. None if off-policy.
        end_of_episode : boolean or None
            Did the iteration reach the end of an episode? None if off-policy.
        """
        actor_critic = ActorCritic()
        actor_critic.copy_parameters_from(self.brain.actor_critic)

        trajectory = self.explore(actor_critic) if on_policy else self.buffer.sample(OFF_POLICY_MINIBATCH_SIZE)
        if not trajectory:
            return None, None

        _, _, _, next_states, _, _ = trajectory[-1]
        action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(next_states)))
        retrace_action_value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1)

        for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(states)))
            average_action_probabilities, _ = self.brain.average_actor_critic(Variable(torch.FloatTensor(states)))
            value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1) * torch.from_numpy(1. - done)
            importance_weights = action_probabilities.data / torch.from_numpy(exploration_probabilities)
            action_indices = Variable(torch.LongTensor(actions))

            naive_advantage = action_values.gather(-1, action_indices).data - value

            retrace_action_value = torch.FloatTensor(rewards) \
                                   + DISCOUNT_FACTOR * retrace_action_value * torch.from_numpy(1. - done)
            retrace_advantage = retrace_action_value - value

            actor_loss = - Variable(importance_weights.gather(-1, action_indices.data).clamp(max=TRUNCATION_PARAMETER)
                                    * retrace_advantage) \
                         * action_probabilities.gather(-1, action_indices).log()
            actor_loss += - (Variable((1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) *
                                      naive_advantage * action_probabilities.data)
                             * action_probabilities.log()).sum()
            actor_gradients = torch.autograd.grad(actor_loss.mean(), action_probabilities, retain_graph=True)
            actor_gradients = self.trust_region_update(actor_gradients, action_probabilities,
                                                       Variable(average_action_probabilities.data))
            action_probabilities.backward(actor_gradients, retain_graph=True)

            critic_loss = (action_values.gather(-1, action_indices) - Variable(retrace_action_value)).pow(2)
            critic_loss.mean().backward(retain_graph=True)

            entropy_loss = ENTROPY_REGULARIZATION * (action_probabilities * action_probabilities.log()).sum(-1)
            entropy_loss.mean().backward(retain_graph=True)

            retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) * \
                                   (retrace_action_value - action_values.gather(-1, action_indices).data) + value
        self.brain.actor_critic.copy_gradients_from(actor_critic)
        self.optimizer.step()
        self.brain.average_actor_critic.copy_parameters_from(self.brain.actor_critic, decay=TRUST_REGION_DECAY)

        if on_policy:
            end_of_episode = trajectory[-1].done[0, 0]
            trajectory_rewards = sum([transition.rewards[0, 0] for transition in trajectory])
            return trajectory_rewards, end_of_episode
        else:
            return None, None

    @staticmethod
    def trust_region_update(actor_gradients, action_probabilities, average_action_probabilities):
        """
        Update the actor gradients so that they satisfy a linearized KL constraint with respect
        to the average actor-critic network. See Section 3.3 of the paper for details.

        Parameters
        ----------
        actor_gradients : tuple of torch.Tensor's
            The original gradients.
        action_probabilities
            The action probabilities according to the current actor-critic network.
        average_action_probabilities
            The action probabilities according to the average actor-critic network.

        Returns
        -------
        tuple of torch.Tensor's
            The updated gradients.
        """
        negative_kullback_leibler = - ((average_action_probabilities.log() - action_probabilities.log())
                                       * average_action_probabilities).sum(-1)
        kullback_leibler_gradients = torch.autograd.grad(negative_kullback_leibler.mean(),
                                                         action_probabilities, retain_graph=True)
        updated_actor_gradients = []
        for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
            scale = actor_gradient.mul(kullback_leibler_gradient).sum(-1).unsqueeze(-1) - TRUST_REGION_CONSTRAINT
            scale = torch.div(scale, actor_gradient.mul(actor_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.)
            updated_actor_gradients.append(actor_gradient - scale * kullback_leibler_gradient)
        return updated_actor_gradients

    def explore(self, actor_critic):
        """
        Explore an environment by taking a sequence of actions and saving the results in the memory.

        Parameters
        ----------
        actor_critic : ActorCritic
            The actor-critic model to use to explore.
        """
        state = self.env.env.state
        if isinstance(state, tuple):
            state = np.array(list(state))
        trajectory = []
        for step in range(MAX_STEPS_BEFORE_UPDATE):
            action_probabilities, action_values = actor_critic(Variable(torch.FloatTensor(state)))

            action = action_probabilities.multinomial()
            action = action.data.numpy()
            next_state, reward, done, _ = self.env.step(action[0])
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
            if done:
                self.env.reset()
                break
            else:
                state = next_state
        return trajectory
