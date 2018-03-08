## reference - playing atari with deep RL, David Dilver et. al. 
## reference - https://keon.io/deep-q-learning/
## Solving a simple cartpole problem with a 2 layered deep q network with experience replay

#prerequisites
#using tensorflow backend

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

#open ai gym environment
env = gym.make('CartPole-v0')

#deep q net with 2 hidden layers
class QNetwork:
    def __init__(self, learning_rate = 0.01, state_size = 4, action_size = 2, hidden_size = 4):

        self.model = Sequential()

        self.model.add(Dense(hidden_size, activation = 'relu', input_dim = state_size))

        self.model.add(Dense(hidden_size, activation = 'relu'))

        self.model.add(Dense(action_size, activation = 'linear'))

        self.optimizer = Adam(lr = learning_rate)

        self.model.compile(loss = 'mse', optimizer = self.optimizer)

#defining memory for experience replay
class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size = batch_size, replace = False)
        return [self.buffer[ii] for ii in idx]

train_episodes = 1000 #maximum number of episodes for training
max_steps = 250
gamma = 0.99 #discount factor

# Exploration parameters
explore_prob = 0.05           #  exploration probability

# Network parameters
hidden_size = 16               # number of units in each deep Q-network hidden layer
learning_rate = 0.001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)

## Populating the memory for experience replay

# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())
state = np.reshape(state, [1, 4])

memory = Memory(max_size=memory_size)

# Initialize replay memory
for ii in range(pretrain_length):

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(state, [1, 4])
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

step = 0
for ep in range(1, train_episodes):
    total_reward = 0
    t = 0
    while t < max_steps:
        step += 1
 
        # Selecting a random action with the exlporation probability
        explore_p = explore_prob
        if explore_p > np.random.rand():
            # Make a random action
            action = env.action_space.sample()
        else:
            # Get action from Q-network
            Qs = mainQN.model.predict(state)[0]
            action = np.argmax(Qs)

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward

        if done:
            # the episode ends so no next state
            next_state = np.zeros(state.shape)
            t = max_steps

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(total_reward),
                  'Explore P: {:.4f}'.format(explore_p))

            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
            state = np.reshape(state, [1, 4])
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state
            t += 1

        # Replay
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
		
		#sampling random minibatch of transitions from memory 
        minibatch = memory.sample(batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
			#for terminal and non terminal transitions
			
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target_Q = mainQN.model.predict(next_state_b)[0]
                target = reward_b + gamma * np.amax(mainQN.model.predict(next_state_b)[0])
            targets[i] = mainQN.model.predict(state_b)
            targets[i][action_b] = target
			
			#DQN fit method
        mainQN.model.fit(inputs, targets, epochs=1, verbose=0)
		
