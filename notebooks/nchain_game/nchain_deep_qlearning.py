import gym
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optimizers


class Agent:
    def __init__(self):
        self.learning_rate = 0.05
        self.neural_network = NeuralNetwork(self.learning_rate)
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
                if self.__with_probability(self.epsilon):
                    action = self.__choose_random_action(env)
                else:
                    action = self.__get_action_with_highest_expected_reward(state)
                
                # perform the action
                new_state, reward, end_game, _ = env.step(action)
                
                total_reward += reward
                
                # Train the neural network
                target_output = self.neural_network.predict_expected_rewards_for_each_action(state)
                target_output[action] = reward + self.discount_factor*self.__get_expected_reward_in_next_state(new_state)
                self.neural_network.train(state, target_output)
                
                state = new_state
            
            self.average_reward_for_each_game.append(total_reward / 1000.)
            
            print(tabulate(self.neural_network.results(), showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))
    
    def __with_probability(self, probability):
        return np.random.random() < probability
    
    def __choose_random_action(self, env):
        return env.action_space.sample()
    
    def __get_action_with_highest_expected_reward(self, state):
        return np.argmax(self.neural_network.predict_expected_rewards_for_each_action(state))
            
    def __get_expected_reward_in_next_state(self, next_state):
        return np.max(self.neural_network.predict_expected_rewards_for_each_action(next_state))


class NeuralNetwork(Sequential):
    """
    5 states
    10 neurons
    2 actions
    """
    def __init__(self, learning_rate=0.05):
        super().__init__()
        self.add(InputLayer(batch_input_shape=(1, 5)))
        self.add(Dense(10, activation='sigmoid'))
        self.add(Dense(2, activation='linear'))
        self.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate))
    
    def train(self, state, target_output):
        input_signal = self.__convert_state_to_neural_network_input(state)
        target_output = target_output.reshape(-1, 2)
        self.fit(input_signal, target_output, epochs=1, verbose=0)
    
    def predict_expected_rewards_for_each_action(self, state):
        input_signal = self.__convert_state_to_neural_network_input(state)
        return self.predict(input_signal)[0]
    
    def results(self):
        result = []
        for state in range(5):
            result.append(self.predict_expected_rewards_for_each_action(state))
        
        return result
    
    def __convert_state_to_neural_network_input(self, state):
        input_signal = np.zeros((1, 5))
        input_signal[0, state] = 1
        return input_signal
    

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
