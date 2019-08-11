import pong_utils
import progressbar as pb
import gym
import time
import matplotlib
import matplotlib.pyplot as plt
from parallelEnv import parallelEnv
import numpy as np
import Agent
class Timer:
    def __init__(self, episode):
        # widget bar to display progress
        self.widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]
        self.timer = pb.ProgressBar(widgets=self.widget, maxval=episode).start()
    def display(self, cur_episode, total_rewards):
        # display some progress every 20 iterations
        if (cur_episode+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(cur_episode+1,np.mean(total_rewards)))
            print(total_rewards)
        # update progress widget bar
        self.timer.update(cur_episode+1)
class PlayPong:
    def __init__(self, episode=500, discount_rate=0.99, epsilon=0.1, beta =0.01, tmax=320, epoch=4, concurrent_agent=8, seed=1231):
        self.envs = parallelEnv('PongDeterministic-v4', concurrent_agent, seed)
        self.pong_agent = Agent.PongAgent()
        self.time_display = Timer(episode)
        self.episode = episode
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.beta = beta
        self.tmax = tmax
        self.epoch = epoch
        # keep track of progress
        self.mean_rewards = []
    def train(self):
        for e in range(self.episode):

            # collect trajectories
            old_probs, states, actions, rewards = pong_utils.collect_trajectories(self.envs, self.pong_agent.policy, tmax=self.tmax)

            total_rewards = np.sum(rewards, axis=0)

            self.pong_agent.train(self.epoch, old_probs, states, actions, rewards, epsilon=self.epsilon, beta=self.beta)
            # the clipping parameter reduces as time goes on
            self.epsilon*=.999
            # the regulation term also reduces
            # this reduces exploration in later runs
            self.beta*=.995
            # get the average reward of the parallel environments
            self.mean_rewards.append(np.mean(total_rewards))
            self.time_display.display(e, total_rewards)
        self.time_display.timer.finish()
        torch.save(self.pong_agent.policy, 'PongAgent.policy')
if __name__ == "__main__":
    game = PlayPong()
    game.train()
