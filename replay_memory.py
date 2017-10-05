import random
import numpy as np
from itertools import zip_longest
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'action_probabilities'))


class ReplayBuffer:
    def __init__(self):
        self.episodes = deque([[]], maxlen=25)

    def add(self, transition):
        if self.episodes[-1] and self.episodes[-1][-1].done:
            self.episodes.append([])
        self.episodes[-1].append(transition)

    def sample(self, batch_size):
        trajectories = random.sample(self.episodes, min(batch_size, len(self.episodes)))
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
        action_probabilities = np.ones_like(transition.action_probabilities) \
                               / transition.action_probabilities.shape[-1]
        transition = Transition(states=transition.next_states,
                                actions=transition.actions,
                                rewards=np.array([[0.]], dtype=np.float32),
                                next_states=transition.next_states,
                                done=transition.done,
                                action_probabilities=action_probabilities)
        return transition
