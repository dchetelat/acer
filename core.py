import gym
from gym.spaces import Discrete as DiscreteSpace
from gym.spaces import Box as ContinuousSpace

ENVIRONMENT_NAME = 'MountainCarContinuous-v0'
# ENVIRONMENT_NAME = 'CartPole-v0'

env = gym.make(ENVIRONMENT_NAME)
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

LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 25
TRUNCATION_PARAMETER = 10
DISCOUNT_FACTOR = 0.99
REPLAY_RATIO = 4
MAX_EPISODES = 200
MAX_STEPS_BEFORE_UPDATE = 200
NUMBER_OF_AGENTS = 1
OFF_POLICY_MINIBATCH_SIZE = 16
TRUST_REGION_CONSTRAINT = 1.
TRUST_REGION_DECAY = 0.99
ENTROPY_REGULARIZATION = 1e-1
MAX_REPLAY_SIZE = 20
