import random
import numpy as np
from itertools import zip_longest
from collections import deque, namedtuple
from core import *

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'action_probabilities'))


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
        if self.episodes[-1] and self.episodes[-1][-1].done:
            self.episodes.append([])
        self.episodes[-1].append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of trajectories from the buffer. If they are of unequal length
        (which is likely), the trajectories will be padded with zero-reward transitions.

        Parameters
        ----------
        batch_size : int
            The batch size of the sample.

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
            batch.append(Transition(*map(lambda data: np.vstack(data), zip(*transitions))))
            previous_transitions = transitions
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
        if not transition.done:
            raise ValueError("Can only extend a terminal transition.")
        action_probabilities = np.ones_like(transition.action_probabilities) \
                               / transition.action_probabilities.shape[-1]
        transition = Transition(states=transition.next_states,
                                actions=transition.actions,
                                rewards=np.array([[0.]], dtype=np.float32),
                                next_states=transition.next_states,
                                done=transition.done,
                                action_probabilities=action_probabilities)
        return transition
