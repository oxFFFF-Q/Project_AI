from pommerman import constants

MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_EVERY = 10  # Terminal states (end of episodes)
MAX_BUFFER_SIZE = 100_000
MAX_BUFFER_SIZE_PRE = 1_000_000
DISCOUNT = 0.95
MAX_STEPS = constants.MAX_STEPS
n_step = 5

Threshold_min = -0.40 / 100
Threshold_max = -0.20 / 300

# Environment settings
EPISODES = 100000
SHOW_EVERY = 1

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.95

SHOW_PREVIEW = True
SHOW_GAME = 100
