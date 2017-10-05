import gym
from gym.spaces import Discrete as DiscreteSpace
from gym.spaces import Box as ContinuousSpace


env = gym.make('CartPole-v0')
action_space = env.action_space
state_space = env.observation_space
env.close()
del env

if isinstance(action_space, DiscreteSpace):
    ACTION_SPACE_DIM = action_space.n
    CONTROL = 'discrete'
else:
    ACTION_SPACE_DIM = action_space.shape[0]
    CONTROL = 'continuous'
STATE_SPACE_DIM = state_space.shape[0]

TRUNCATION_PARAMETER = 10
DISCOUNT_FACTOR = 0.99
REPLAY_RATIO = 4
MAX_EPISODES = 200
MAX_STEPS_BEFORE_UPDATE = 20
NUMBER_OF_AGENTS = 2
OFF_POLICY_MINIBATCH_SIZE = 16
