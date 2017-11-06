import gym
from gym.spaces import Discrete as DiscreteSpace

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

# Parameters that work well for MountainCarContinuous-v0
LEARNING_RATE = 5e-4
REPLAY_BUFFER_SIZE = 25
TRUNCATION_PARAMETER = 5
DISCOUNT_FACTOR = 0.99
REPLAY_RATIO = 4
MAX_EPISODES = 200
MAX_STEPS_BEFORE_UPDATE = 200
NUMBER_OF_AGENTS = 12
OFF_POLICY_MINIBATCH_SIZE = 4
TRUST_REGION_CONSTRAINT = 1.
TRUST_REGION_DECAY = 0.99
ENTROPY_REGULARIZATION = 0.
MAX_REPLAY_SIZE = 200
NOISE_SCALE = 5
VISUAL = False
USE_ORNSTEIN_UHLENBECK = True
INITIAL_STANDARD_DEVIATION = 5.
ACTOR_LOSS_WEIGHT = 0.1
