# Architecture

We were able to implement all the descendants of Rainbow DQN except Categorical DQN

- [x] basic DQN
- [x] Double DQN
- [x] Prioritized Experience Replay
- [x] Dueling Network Architectures
- [x] Noisy Nets
- [x] Multi Step Reinforcement Learning
- [ ] Categorical DQN

# Getting Started with our DQNAgent

# Pre-requisites

* [Python 3.6.0](https://www.python.org/downloads/release/python-360/)+ (including `pip`)
* [Docker](https://www.docker.com/) (only needed for `DockerAgent`)
* [tensorflow 2.6.0](https://www.tensorflow.org/hub/installation)
* [Keras 2.6.0](https://keras.io/getting_started/)
* Others are all included in [requirements](DQNAgent/requirements.txt)
# Installation

* Clone the repository
```
$ git clone https://github.com/oxFFFF-Q/Project_AI.git
```

## Pip

* Install the `pommerman` package. This needs to be done every time the code is updated to get the
latest modules
```
$ cd ~/playground
$ pip install -U .
```

## Conda

* Install the `pommerman` environment.
```
$ cd ~/playground
$ conda env create -f env.yml
$ conda activate pommerman
```

* To update the environment
```
$ conda env update -f env.yml --prune
```

# Launch the agent
We have seperately trained models for player 1 [Agent1](DQNAgent/agents/Agent1.py) and player 3 [Agent3](DQNAgent/agents/Agent3.py). Run [main_test.py](DQNAgent/main_test.py) to test them palying against two [SimpleAgent](pommerman/agents/simple_agent.py).

# Train your agent

## A Simple Example

Run [main_train.py](DQNAgent/main_train.py) to train our final DQN model for radio team competition of two [SimpleAgent](pommerman/agents/simple_agent.py) as enemies and a [SimpleAgent](pommerman/agents/simple_agent.py) as teammate.

The training will not automatically stop, but need to be done manully, according to the given out report about the rewards. The paramaters will be recorded every 100 episodes. Run [main_save_model.py](DQNAgent/main_save_model.py) to save the model. The name of the model is required. The best one is usually among the last few models.

## Use other strategies

Select other names for `strategy` in [main_train.py](DQNAgent/main_train.py) to try other achietectures. Make sure of the consistency of the `strategy` in [main_save_model.py](DQNAgent/main_save_model.py).



# Visualize the experiment results

Our experiment results are all stored in [data](DQNAgent/result_image/data). Run [make_image.py](DQNAgent/result_image/make_image.py) to get a visualization of them.

# Reference

Our approch get inspired from [Sentdex tutorial](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/)
