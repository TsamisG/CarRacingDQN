import numpy as np
import gym
import cv2
import collections
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
============================================
            Frame Preprocessing
============================================
'''
class Step_Reset(gym.Wrapper):
  def __init__(self, env=None, repeat=4):
    super(Step_Reset, self).__init__(env)
    self.repeat = repeat

  def step(self, action):
    done = False
    step_reward = 0
    for i in range(self.repeat):
      s, r, done, trun, info = self.env.step(action)
      step_reward += r
    return s, step_reward, done, trun, info

  def reset(self):
    s, info = self.env.reset()
    for i in range(35):
      s, _, _, _, _ = self.env.step(0)
    return s, info


class Preprocess_Frames(gym.ObservationWrapper):
  def __init__(self, shape, env=None):
    super(Preprocess_Frames, self).__init__(env)
    self.shape = (shape[2], shape[0], shape[1])
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

  def observation(self, obs):
    obs = obs[:84,6:90,:]
    obs = cv2.convertScaleAbs(obs)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return obs / 255.0


class Stack_Frames(gym.ObservationWrapper):
  def __init__(self, env=None, repeat=4):
    super(Stack_Frames, self).__init__(env)
    self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                            env.observation_space.high.repeat(repeat, axis=0),
                                            dtype=np.float32)
    self.stack = collections.deque(maxlen=repeat)

  def reset(self):
    self.stack.clear()
    obs = self.env.reset()[0]
    for _ in range(self.stack.maxlen):
      self.stack.append(obs)
    return np.array(self.stack).reshape(self.observation_space.low.shape)

  def observation(self, obs):
    self.stack.append(obs)
    return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env, shape=(84, 84, 1), repeat=4):
  env = Step_Reset(env, repeat)
  env = Preprocess_Frames(shape, env)
  env = Stack_Frames(env, repeat)
  return env


'''
============================================
            Memory Object
============================================
'''

class Memory():
  def __init__(self, memory_size, input_shape, n_actions):
    self.memory_size = memory_size
    self.input_shape = input_shape
    self.n_actions = n_actions
    self.counter = 0
    self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
    self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
    self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
    self.state2_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
    self.done_memory = np.zeros(self.memory_size, dtype=np.bool_)

  def remember(self, s, a, r, s2, done):
    idx = self.counter % self.memory_size
    self.state_memory[idx] = s
    self.action_memory[idx] = a
    self.reward_memory[idx] = r
    self.state2_memory[idx] = s2
    self.done_memory[idx] = done
    self.counter += 1

  def sample_batch(self, batch_size):
    current_memory_size = min(self.counter, self.memory_size)
    batch_indices = np.random.choice(current_memory_size, batch_size, replace = False)
    states = self.state_memory[batch_indices]
    actions = self.action_memory[batch_indices]
    rewards = self.reward_memory[batch_indices]
    states2 = self.state2_memory[batch_indices]
    dones = self.done_memory[batch_indices]

    return states, actions, rewards, states2, dones


'''
============================================
            Deep Q-Network Object
============================================
'''


class DeepQNetwork(nn.Module):
  def __init__(self, input_dims, n_actions, learning_rate):
    super(DeepQNetwork, self).__init__()

    self.conv1 = nn.Conv2d(input_dims[0], 16, 8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

    fc_input_dims = self.calculate_fc_input_dims(input_dims)

    self.fc1 = nn.Linear(fc_input_dims, 256)
    self.fc2 = nn.Linear(256, n_actions)

    self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def calculate_fc_input_dims(self, input_dims):
    X = T.zeros(1, *input_dims)
    X = self.conv1(X)
    X = self.conv2(X)
    return int(np.prod(X.size()))

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.relu(self.conv2(X))
    X = X.view(X.size()[0], -1)
    X = F.relu(self.fc1(X))
    X = self.fc2(X)

    return X


'''
============================================
            Agent Object
============================================
'''

class DQNAgent():
  def __init__(self, input_shape, n_actions, gamma=0.99, epsilon=1.0, learning_rate=1e-4, memory_size=500,
               batch_size=32, epsilon_min=0.02, epsilon_decay=8e-3, replace_target_every=5000):
    self.input_shape = n_actions
    self.n_actions = n_actions
    self.gamma = gamma
    self.epsilon = epsilon
    self.learning_rate = learning_rate
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.replace_target_every = replace_target_every
    self.action_space = [i for i in range(self.n_actions)]
    self.learn_step_counter = 0

    self.memory = Memory(memory_size, input_shape, n_actions)

    self.Qeval_Network = DeepQNetwork(input_shape, n_actions, learning_rate)

    self.Qtarget_Network = DeepQNetwork(input_shape, n_actions, learning_rate)

  def choose_action(self, state):
    if np.random.random() > self.epsilon:
      state = T.tensor(np.array([state]), dtype=T.float).to(self.Qeval_Network.device)
      Qvalues = self.Qeval_Network.forward(state)
      action = T.argmax(Qvalues).item()
    else:
      action = np.random.choice(self.action_space)
    return action

  def remember(self, s, a, r, s2, done):
    self.memory.remember(s, a, r, s2, done)

  def sample_from_memory(self):
    states, actions, rewards, states2, dones = self.memory.sample_batch(self.batch_size)
    states = T.tensor(states).to(self.Qeval_Network.device)
    actions = T.tensor(actions).to(self.Qeval_Network.device)
    rewards = T.tensor(rewards).to(self.Qeval_Network.device)
    states2 = T.tensor(states2).to(self.Qeval_Network.device)
    dones = T.tensor(dones).to(self.Qeval_Network.device)

    return states, actions, rewards, states2, dones

  def update_target_network(self):
    if self.learn_step_counter % self.replace_target_every == 0:
      self.Qtarget_Network.load_state_dict(self.Qeval_Network.state_dict())

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

  def learn(self):
    if self.memory.counter < self.batch_size:
      return

    self.Qeval_Network.zero_grad()
    self.update_target_network()

    states, actions, rewards, states2, dones = self.sample_from_memory()

    indices = np.arange(self.batch_size)

    Qhat = self.Qeval_Network(states)[indices, actions]
    Qhat2 = self.Qtarget_Network(states2).max(dim=1)[0]

    Qhat2[dones] = 0.0

    G = rewards + self.gamma*Qhat2

    J = self.Qeval_Network.loss(G, Qhat).to(self.Qeval_Network.device)
    J.backward()
    self.Qeval_Network.optimizer.step()
    self.learn_step_counter += 1
    self.decay_epsilon()

  def create_checkpoint(self, path):
    T.save(self.Qeval_Network.state_dict(), path+'Qeval_Network.pt')
    T.save(self.Qtarget_Network.state_dict(), path+'Qtarget_Network.pt')
    np.save(path+'state_memory.npy', self.memory.state_memory)
    np.save(path+'action_memory.npy', self.memory.action_memory)
    np.save(path+'reward_memory.npy', self.memory.reward_memory)
    np.save(path+'state2_memory.npy', self.memory.state2_memory)
    np.save(path+'done_memory.npy', self.memory.done_memory)

  def load_checkpoint(self, path):
    self.Qeval_Network.load_state_dict(T.load(path+'Qeval_Network.pt'))
    self.Qtarget_Network.load_state_dict(T.load(path+'Qtarget_Network.pt'))
    self.memory.state_memory = np.load(path+'state_memory.npy')
    self.memory.action_memory = np.load(path+'action_memory.npy')
    self.memory.reward_memory = np.load(path+'reward_memory.npy')
    self.memory.state2_memory = np.load(path+'state2_memory.npy')
    self.memory.done_memory = np.load(path+'done_memory.npy')