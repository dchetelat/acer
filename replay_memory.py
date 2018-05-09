import random
import torch
from collections import deque, namedtuple
from core import *

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'exploration_statistics'))


class ReplayBuffer:
    """
    Replay buffer for the agents.
    """
    def __init__(self):
        self.episodes = deque([[]], maxlen=REPLAY_BUFFER_SIZE)

    def add(self, transition):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        transition : Transition
            The transition to add.
        """
        if self.episodes[-1] and self.episodes[-1][-1].done[0, 0]:
            self.episodes.append([])
        self.episodes[-1].append(transition)

    def sample(self, batch_size, window_length=float('inf')):
        """
        Sample a batch of trajectories from the buffer. If they are of unequal length
        (which is likely), the trajectories will be padded with zero-reward transitions.

        Parameters
        ----------
        batch_size : int
            The batch size of the sample.
        window_length : int, optional
            The window length.

        Returns
        -------
        list of Transition's
            A batched sampled trajectory.
        """
        batched_trajectory = []
        trajectory_indices = random.choices(range(len(self.episodes)-1), k=min(batch_size, len(self.episodes)-1))
        trajectories = []
        for trajectory in [self.episodes[index] for index in trajectory_indices]:
            start = random.choices(range(len(trajectory)), k=1)[0]
            trajectories.append(trajectory[start:start + window_length])
        smallest_trajectory_length = min([len(trajectory) for trajectory in trajectories]) if trajectories else 0
        for index in range(len(trajectories)):
            trajectories[index] = trajectories[index][-smallest_trajectory_length:]
        for transitions in zip(*trajectories):
            batched_transition = Transition(*[torch.cat(data, dim=0) for data in zip(*transitions)])
            batched_trajectory.append(batched_transition)
        return batched_trajectory

    @staticmethod
    def extend(transition):
        """
        Generate a new zero-reward transition to extend a trajectory.

        Parameters
        ----------
        transition : Transition
            A terminal transition which will become the new transition's previous
            transition in the trajectory.

        Returns
        -------
        Transition
            The new transition that can be used to extend a trajectory.
        """
        if not transition.done[0, 0]:
            raise ValueError("Can only extend a terminal transition.")
        exploration_statistics = torch.ones(transition.exploration_statistics.size()) \
                                 / transition.exploration_statistics.size(-1)
        transition = Transition(states=transition.next_states,
                                actions=transition.actions,
                                rewards=torch.FloatTensor([[0.]]),
                                next_states=transition.next_states,
                                done=transition.done,
                                exploration_statistics=exploration_statistics)
        return transition
