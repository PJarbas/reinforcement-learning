import gym
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 2))
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.5
        self.decay_factor = 0.999
        self.average_reward_for_each_game = []
         
    def play(self, env, number_of_games=200):
        
        for game in range(number_of_games):
            
            print(f"Game {game} of {number_of_games}")
            
            # reset the environmet before play begins   
            state = env.reset()
            
            self.epsilon *= self.decay_factor
            
            total_reward = 0
            
            # play the game until it ends after 1000 steps
            end_game = False
            while not end_game:
                
                # choose action 0 or 1 with 50% of probability
                if self.__q_table_is_empty(state) or self.__with_probability(self.epsilon):
                    action = self.__choose_random_action(env)
                else:
                    action = self.__get_action_with_highest_expected_reward(state)
                
                # perform the action
                new_state, reward, end_game, _ = env.step(action)
                
                total_reward += reward
                
                # Q learning equation
                self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor*self.__get_expected_reward_in_next_state(new_state) - self.q_table[state, action])
                
                state = new_state
            
            self.average_reward_for_each_game.append(total_reward / 1000.)
            
            print(tabulate(self.q_table, showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

        
    def __q_table_is_empty(self, state):
        return sum(self.q_table[state, :] == 0)
    
    def __with_probability(self, probability):
        return np.random.random() < probability
    
    def __choose_random_action(self, env):
        return env.action_space.sample()
    
    def __get_action_with_highest_expected_reward(self, state):
        return np.argmax(self.q_table[state, :])
            
    def __get_expected_reward_in_next_state(self, next_state):
        return np.max(self.q_table[next_state, :])


def graph_average_reward(average_reward):
    plt.plot(average_reward)
    plt.title("Performance over Time")
    plt.ylabel("Average Reward")
    plt.xlabel("Games")
    plt.show()
        
# create environment 
env = gym.make('NChain-v0')

agent = Agent()

agent.play(env)

# plot the results
graph_average_reward(agent.average_reward_for_each_game)
