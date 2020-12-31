# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 01:35:13 2021

@author: mridul
"""
import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import pandas as pd
import math

def get_stock_data(stock_file):
    df = pd.read_csv(stock_file + '.csv')
    return list(df['Open']),df['Volume_(BTC)'][0] #Consider Opening price at each minute and initial bitcoint the agent will have
# We assume that the agent bys and sell the bitcoin on the opening price of the minute as sata is of 1 min interval


def evaluate_model(agent, data, window_size,init_xbt,init_amt):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

    state = get_state(data, 0, window_size + 1)# state is the vector with previos window size day records and their relation  we start with t = 0

    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        action = agent.act(state, is_eval=True)# select an action

        # BUY
        if action == 1 and init_amt > 0:
            print("Bought 1 XBT at price :{}".format(format_position(data[t])))
            init_xbt = init_xbt + 1 #We buy 1 bitcoin at that rate
            init_amt -= data[t] # anount deducted from balance
            agent.inventory.append(data[t])
            
            history.append((data[t], "BUY"))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = min(agent.inventory)  # We sell the bitcoin first that we bought at the lowest price
            agent.inventory.pop(np.argmin(agent.inventory))
            init_xbt = init_xbt / 2 # Whenever we have to sell we sell half the coin we own at that moment
            delta = data[t] - bought_price
            reward = delta 
            print("Sold {} XBT at price :{}   || Profit : {}".format(init_xbt,format_position(data[t]),format_position(delta*init_xbt)))
            total_profit += delta*(init_xbt) # Track of total profit
            init_amt += data[t]*(init_xbt) # porfit amount is added in the account balance

            history.append((data[t], "SELL"))
        # HOLD
        else:
            print("Holding") 
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1) # done true if traversed through all data
        agent.memory.append((state, action, reward, next_state, done)) # store this information in memory of the agent

        state = next_state 
        if done:
            return total_profit,init_amt,init_xbt,history


def get_state(data, t, n_days): # here t is the current t and n is the window size
    d = t - n_days + 1 # if we have n previos day records they are stored else the first day record is assumed to be the record of the previous days
    if d >= 0:
      block = data[d: t + 1] 
    else:
      block = -d * [data[0]] + data[0: t + 1]
    #block is now the data for the last window_size days
    res = []
    for i in range(n_days - 1):
        res.append((block[i + 1] - block[i]))
     #res tells us the relation of the data from the last day
    return np.array([res]) #Returns an n-day state representation ending at time t


def show_eval_result(model_name, profit,amt,init_amt,xbt,init_xbt):
    format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))
    print("\nSUMMARY:")
    print('Initial Bank Balance: {}'.format(format_position(init_amt)))
    print('Initial Bitcoins owned(1%): {}'.format(init_xbt))
    print('\n')
    print('Final Bank Balace: {}'.format(format_position(amt)))
    print('Final Bitcoins owned: {}'.format(xbt))
    print('\nTotal Profit: {}'.format(format_position(profit)))
    print('(Trained Model  :{})'.format(model_name))

class Agent:

    def __init__(self, state_size, model_name=None):
      
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)

        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0  # for epsilon greedy algorithm
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))# Add the data in memory of the agent

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:#if we are training the model
            return random.randrange(self.action_size)# take random action in order to diversify experience at the beginning

        action_probs = self.model.predict(state)# if we are predicting we ask the model to predict the value of all possible actions
        #print("action_probs",action_probs)
        return np.argmax(action_probs[0])# we take the option with highest probabilty predicted by the model

    def train_experience(self, batch_size):#Train on previous experiences in memory
      
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward #reward of state action
            else:
              # approximate deep q-learning equation
              # Updating network weights using the Bellman Equation
              q_values = self.model.predict(state)# estimate q-values based on current state
              q_values[0][action] = target# update the target for current action based on discounted reward

              X_train.append(state[0])
              y_train.append(q_values[0])

 
        #parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:#by this in each batch we reduce the epsilon value and when the model is trained few times we will  start predict action from model instead of an random no in method act 
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("model{}".format(episode)) # save the trained model

    def load(self):
        return load_model(self.model_name, custom_objects=self.custom_objects) # load an existing trained model


eval_stock = input("Enter Stock name")
model_name = input("Enter model name")
window_size = 10
data,init_xbt = get_stock_data(eval_stock)
init_xbt = init_xbt*0.01 # initial amount of coins (1%)
init_amt = int(input("Enter initial account balance ( > 100000 )"))
agent1 = Agent(window_size, model_name=model_name) # create an object of the class agent
profit,amt, xbt,_ = evaluate_model(agent1, data, window_size,init_xbt,init_amt) # predictiong from the model
show_eval_result(model_name, profit,amt,init_amt,xbt,init_xbt) # displaying summary