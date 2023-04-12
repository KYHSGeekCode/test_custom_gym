import random
from collections import deque
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.stack(list(map(torch.from_numpy, batch.state)))
        action = torch.stack([torch.Tensor([float(x)]) for x in batch.action])
        reward = torch.stack([torch.Tensor([float(x)]) for x in batch.reward])
        next_state = torch.stack(list(map(torch.from_numpy, batch.next_state)))
        done = torch.stack([torch.Tensor([x]) for x in batch.done])
        return state, action, next_state, reward, done  # tuple(map(torch.cat, batch))


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        buffer_size=30,
        batch_size=32,
        gamma=0.99,
        learning_rate=0.001,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        q_values = self.policy_net(state).gather(1, action.to(torch.int64)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
