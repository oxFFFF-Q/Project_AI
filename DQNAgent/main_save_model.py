def main(model_name='new_model', strategy='DQN_basic'):
    # strategies: 'DQN_basic', 'DQN_double', 'DQN_dueling', 'DQN_priority', 'DQN_noisy', 'DQN_multi_steps', 'DQN_final'

    if strategy == 'DQN_basic':
        from agents.DQNAgent_basic import DQNAgent
    elif strategy == 'DQN_double':
        from agents.DQNAgent_double_dqn import DQNAgent
    elif strategy == 'DQN_dueling':
        from agents.DQNAgent_dueling_dqn import DQNAgent
    elif strategy == 'DQN_priority':
        from agents.DQNAgent_priority_memory import DQNAgent
    elif strategy == 'DQN_noisy':
        from agents.DQNAgent_noisy import DQNAgent
    elif strategy == 'DQN_multi_steps':
        from agents.DQNAgent_multi_steps import DQNAgent
    elif strategy == 'DQN_final':
        from agents.DQNAgent_final import DQNAgent

    agent1 = DQNAgent()

    # load saved weights from checkpoint
    weight = "episode100"
    agent1.load_weights(weight)

    # save your model
    agent1.save_model(model_name)


if __name__ == '__main__':
    # rename your model
    main(model_name='new_model', strategy='DQN_basic')
    # strategies: 'DQN_basic', 'DQN_double', 'DQN_dueling', 'DQN_priority', 'DQN_noisy', 'DQN_multi_steps', 'DQN_final'
