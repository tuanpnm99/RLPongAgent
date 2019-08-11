##This code is adapted from Udacity Deep Reinformance Learning Nano Degree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pong_utils import device

RIGHT=4
LEFT=5
class PongAgent():
    def __init__(self):
        self.policy = Policy().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
    # convert states to probability, passing through the policy
    def states_to_prob(self, states):
        states = torch.stack(states)
        policy_input = states.view(-1,*states.shape[-3:])
        return self.policy(policy_input).view(states.shape[:-3])
    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def clipped_surrogate(self, old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):

        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = self.states_to_prob(states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

        # ratio for clipping
        ratio = new_probs/old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))


        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta*entropy)
    def train(self, epoch, old_probs, states, actions, rewards, epsilon=0.1, beta=0.01):
        # gradient ascent step
        for _ in range(epoch):
            L = -self.clipped_surrogate(old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
