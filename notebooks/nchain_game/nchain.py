import gym
from tabulate import tabulate
import numpy as np

class Agent:
    def __init__(self):
        self.reward_table = np.zeros((5, 2))
        
    def play(self, env):
        
        # reset the environmet before play begins   
        state = env.reset()
        
        # play the game until it ends after 1000 steps
        end_game = False
        while not end_game:
            
            # choose action 0 or 1 with 50% of probability
            if self.__reward_table_is_empty(state):
                action = self.__choose_random_action(env)
            else:
                action = self.__get_action_with_highest_expected_reward(state)
            
            # perform the action
            new_state, reward, end_game, _ = env.step(action)
            
            self.reward_table[state, action] += reward
            
            print(f"Starting in state {state} Took action {action}, Entered new state {new_state} and receive reward {reward}")
            print(tabulate(self.reward_table, showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))
            
            state = new_state
    
    def __reward_table_is_empty(self, state):
        return sum(self.reward_table[state, :] == 0)
    
    def __choose_random_action(self, env):
        return env.action_space.sample()
    
    def __get_action_with_highest_expected_reward(self, state):
        return np.argmax(self.reward_table[state, :])
            

# create environment 
env = gym.make('NChain-v0')

agent = Agent()

agent.play(env)
