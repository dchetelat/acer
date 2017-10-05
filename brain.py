import torch
import torch.nn.functional as F
from core import *


class ActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = torch.nn.Linear(STATE_SPACE_DIM, 32)
        self.hidden_layer = torch.nn.Linear(32, 32)
        self.action_layer = torch.nn.Linear(32, ACTION_SPACE_DIM)
        if CONTROL is 'discrete':
            self.action_value_layer = torch.nn.Linear(32, ACTION_SPACE_DIM)

    def forward(self, states):
        hidden = F.relu(self.input_layer(states))
        hidden = F.relu(self.hidden_layer(hidden))
        if CONTROL is 'discrete':
            action_probabilities = F.softmax(self.action_layer(hidden))
            action_values = self.action_value_layer(hidden)
            return action_probabilities, action_values

    def copy_parameters_from(self, source, decay=0.):
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            # parameter.data.copy_(decay * parameter.data + (1 - decay) * source_parameter.data)
            parameter.data.copy_(decay * parameter.data + (1 - decay) * source_parameter.data)

    def copy_gradients_from(self, source):
        for parameter, source_parameter in zip(self.parameters(), source.parameters()):
            # parameter.grad.data.copy_(source_parameter.grad.data)
            parameter._grad = source_parameter.grad

# brain_actor_critic = ActorCritic()
# brain_actor_critic.share_memory()


class Brain:
    def __init__(self):
        self.actor_critic = ActorCritic()
        self.actor_critic.share_memory()

brain = Brain()
