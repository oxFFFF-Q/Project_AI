# Getting Started with our DQNAgent

# Pre-requisites

* [Python 3.6.0](https://www.python.org/downloads/release/python-360/)+ (including `pip`)
* [Docker](https://www.docker.com/) (only needed for `DockerAgent`)
* [tensorflow 2.6.2](https://www.tensorflow.org/hub/installation)
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

# Train your agent

## A Simple Example

Run [simple_ffa_run.py](../examples/simple_ffa_run.py) to train our final DQN model for radio team competition of two
[SimpleAgent](../pommerman/agents/simple_agent.py) as enemies and a same DQNAgent as teammate on the board.

## Train other models

The above example can be extended to use [DockerAgent](../pommerman/agents/docker_agent.py) instead of a
[RandomAgent](../pommerman/agents/random_agent.py). [examples/docker-agent](../examples/docker-agent) contains
the code to wrap a [SimpleAgent](../pommerman/agents/simple_agent.py) inside Docker.


* We will build a docker image with the name "pommerman/simple-agent" using the `Dockerfile` provided.
```
$ cd ~/playground
$ docker build -t pommerman/simple-agent -f examples/docker-agent/Dockerfile .
```

* The agent list seen in the previous example can now be updated. Note that a `port` argument (of an unoccupied port) is
needed to expose the HTTP server.
```python
agent_list = [
    agents.SimpleAgent(),
    agents.RandomAgent(),
    agents.SimpleAgent(),
    agents.DockerAgent("pommerman/simple-agent", port=12345)
]
```

# Launch the agent

# Visualizing the experiment results
