import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'terminal', 'aborted', 'action_probabilities'))


class ReplayBuffer:
    def __init__(self):
        self.episodes = deque([[]], maxlen=25)

    def add(self, transition):
        if self.episodes[-1] and (self.episodes[-1][-1].terminal or self.episodes[-1][-1].aborted):
            self.episodes.append([])
        self.episodes[-1].append(transition)

    def sample(self, max_length=float('inf')):
        episode = self.episodes[random.randrange(len(self.episodes))]
        if len(episode) <= max_length:
            return episode
        else:
            start = random.randrange(len(episode) - max_length)
            return episode[start: start + max_length]
