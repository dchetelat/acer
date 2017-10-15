import random
import torch
from itertools import zip_longest
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

    def sample(self, batch_size, max_length=float('inf')):
        """
        Sample a batch of trajectories from the buffer. If they are of unequal length
        (which is likely), the trajectories will be padded with zero-reward transitions.

        Parameters
        ----------
        batch_size : int
            The batch size of the sample.
        max_length : int, optional
            The maximum length of the batch. Longer batches will be randomly truncated
            to fit this length.

        Returns
        -------
        list of Transition's
            A batched sampled trajectory.
        """
        trajectory_indices = random.sample(range(len(self.episodes)-1), min(batch_size, len(self.episodes)-1))
        trajectories = [self.episodes[index] for index in trajectory_indices]
        batch = []
        previous_transitions = tuple([None for _ in range(batch_size)])
        for transitions in zip_longest(*trajectories, fillvalue=None):
            transitions = [transition if transition else self.extend(previous_transition)
                           for transition, previous_transition in zip(transitions, previous_transitions)]
            batch.append(Transition(*map(lambda data: torch.cat(data, dim=0), zip(*transitions))))
            previous_transitions = transitions
        if len(batch) > max_length:
            start = random.randrange(len(batch) - max_length)
            batch = batch[start:start+max_length]
        return batch

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
