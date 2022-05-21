import gym
import numpy as np
import math


class MountainCar():
    def __init__(self, buckets=(6, 12), num_episodes=1500, min_lr=0.1, min_explore=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_explore = min_explore
        self.discount = discount
        self.decay = decay
        self.env = gym.make('MountainCar-v0')
        self.upper_bounds = [self.env.observation_space.high[0],self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0],self.env.observation_space.low[1]]
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def get_explore_rate(self, t):
        return max(self.min_explore, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_lr(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def update_q(self, state, action, reward, new_state,n):
        self.Q_table[new_state][action] = self.Q_table[state][action] + 1/n * (reward - self.Q_table[state][action])

    def SARSA_update_q(self, state, action, reward, new_state, new_action):
        self.Q_table[state][action] = self.Q_table[state][action] + self.lr * (reward + self.discount * self.Q_table[new_state][new_action] - self.Q_table[state][action])

    def monte_update_q(self, returns,state,action):
        self.Q_table[state][action]=sum(returns[(state,action)])/len(returns[(state,action)])


    def choose_action(self, state):
        if np.random.uniform() < self.explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)


    def q_train(self):
        new_state=0
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())
            self.lr = self.get_lr(e)
            self.explore_rate = self.get_explore_rate(e)
            done = False
            r=0
            n=0
            action = self.choose_action(current_state)
            while not done:
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                n+=1
                r+=reward
                action = self.choose_action(new_state)

            self.update_q(current_state, action, r, new_state,n)

        print('Finished training!')

    def SARSA_train(self):

        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())
            self.lr = self.get_lr(e)
            self.explore_rate = self.get_explore_rate(e)
            done = False
            action = self.choose_action(current_state)

            while not done:
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.choose_action(new_state)
                self.SARSA_update_q(current_state, action, reward, new_state,new_action)
                action = new_action
                current_state=new_state

        print('Finished training!')

    def monte_carlo_train(self):
        returns=dict()

        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())
            self.explore_rate = self.get_explore_rate(e)

            action = self.choose_action(current_state)
            self.lr = self.get_lr(e)
            done = False
            r=0
            while not done:
                obs, reward, done, _ = self.env.step(action)
                r+=reward
                if returns.get((current_state,action)):
                    returns[(current_state,action)].append(r)
                else:
                    returns[(current_state,action)]=[r]

                current_state=self.discretize_state(obs)
                action = self.choose_action(current_state)

            if returns.get((current_state, action)):
                self.monte_update_q(returns,current_state,action)


        print('Finished training!')



    def run(self,run_episodes=5):
        acc = 0
        for e in range(run_episodes):
            current_state = self.discretize_state(self.env.reset())
            done = False
            for i in range(200):
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                self.env.render()
                current_state = self.discretize_state(obs)
                if obs[0]>=0.5:
                    acc+=1
                    break
        print(acc/run_episodes)




        print('Finished running!')
        # Write your code here


agent = MountainCar()

agent.SARSA_train()
agent.run(20)
