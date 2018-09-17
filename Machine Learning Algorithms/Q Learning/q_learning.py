# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 2018

@author: Xuan Yang

REF:
    http://mnemstudio.org/path-finding-q-learning-tutorial.htm
    http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
    
"""
import numpy as np

"""
    Q learning algorithm:
        type: Reinforcement Learning
        parameters:
            r_matrix: an 2d matrix contain information such as states and actions as well as instant reword.
            gamma: learning rate, default is 0.8
            ite_count: training iteration count
        return value:
            2D matrix with same structre as the R-matrix provided.
        core idea: 
            Reward matrix(R-matrix):
                What is it: A 2D matrix made with states and actions
                Purpose: record the instant reward for taking each 
                    action in each state
            Q-matrix:
                What is it: A 2D matrix made with states and actions
                Purpose: Used to track the estimate reward for taking each 
                    action in each state
                Initialize: The matrix is initialized to zero
                Update: 
                    Q(state, action) = R(state, action) + Gamma * Max(Q(next_state, next_action))
                    
                    At certain state the expected reward of an action is 
                    the sum of the immediate reward of the action plus the 
                    maximum of the expected reward at next state.
                
        algorithm:
            1. initialize reward and Q matrix
            2. training:
                a. generate a random starting state and action
                b. update Q-matrix using the formula
                c. repeat a and b until iteration count reached
            3. return Q-matrix
"""

class Q_Learning:
    
    def __init__(self, r_matrix, gamma = 0.8, ite_count = 100):
        self.r_matrix = np.matrix(r_matrix)
        self.gamma = gamma
        self.ite_count = ite_count
        self.q_matrix = np.matrix(np.zeros(self.r_matrix.shape)) #create a q matrix with the same structure as r matrix and initialized to 0
        
        """
        main training method
        """
    def __train(self, state, action):
        #get max expected reword at next state
        next_state = action                                 #next state is the action we take at current state
        max_q = np.max(self.q_matrix[next_state])
        
        #Q-matrix update formula
        self.q_matrix[state, action] = self.r_matrix[state, action] + self.gamma * max_q

        
    """
    main loop method
    """
    def __learning(self):
        for i in range(self.ite_count):
            #generate randem initial state
            state = np.random.randint(0, self.r_matrix.shape[0])
            
            #get all availiable action in current state and choose one randomly
            actions = np.where(self.r_matrix[state] >= 0)[1]
            action = np.random.choice(actions)
            
            #call training function with state and action
            self.__train(state, action)

    
    def get_q_table(self):
        self.__learning()
        return self.q_matrix/np.max(self.q_matrix)*100      #normalize the table
    



