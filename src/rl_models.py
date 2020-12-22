import random
import math
import numpy as np


# exploration first
class Explfirst_mab:    
    
    def __init__(self, num_arms, name="mab"):
        self.name = name
        self.num_arms = num_arms
        self.estimated_reward = [-1] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self):  
        
        # print("*** iteration #%d ***" % (self.iteration))
        
        rand = random.uniform(0, 1)
        
        # exploit by default
        action_ix = np.argmax(self.estimated_reward)
        
        # explore unexplored actions if any (with reward -1)
        if -1 in self.estimated_reward:
            unexplored_action_ixs = [i for i, x in enumerate(self.estimated_reward) if x == -1]
            # if array not empty, pick random from unexplored actions
            if unexplored_action_ixs:
                action_ix = random.choice(unexplored_action_ixs)
        
        #print("   - action_ix: %d" % action_ix)
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] = reward_last_action
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
      

        
# epsilon greedy
class Epsilongreedy_mab:    
    
    def __init__(self, epsilon_original, num_arms, name="mab"):
        self.epsilon_original = epsilon_original
        self.name = name
        self.epsilon = epsilon_original
        self.num_arms = num_arms
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self):  
        
        #print("*** iteration #%d ***" % (self.iteration))
        
        rand = random.uniform(0, 1)
        
        #print("select_action: rand = %.2f, eps = %.2f" % (rand,self.epsilon))
                
        # explore
        if rand < self.epsilon:
            #print("   - explore")
            action_ix = random.randint(0, self.num_arms - 1)
                    
        # exploit best action
        else:
            #print("   - exploit")
            action_ix = np.argmax(self.estimated_reward)
            
        #print("   - action_ix: %d" % action_ix)
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        self.update_epsilon(self.epsilon_original, self.iteration)
        
    def update_epsilon(self,epsilon_original, iteration):
        self.epsilon = epsilon_original / math.sqrt(iteration)
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
# stateless Q-learning
class Stateless_qlearning:    
    
    def __init__(self, epsilon_original, alpha, gamma, num_arms, name="mab"):
        self.epsilon_original = epsilon_original
        self.alpha = alpha
        self.gamma = gamma
        self.name = name
        self.epsilon = epsilon_original
        self.num_arms = num_arms
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self):  
        
#         print("*** iteration #%d ***" % (self.iteration))
        
        rand = random.uniform(0, 1)
        
#         print("select_action: rand = %.2f, eps = %.2f" % (rand,self.epsilon))
                
        # explore
        if rand < self.epsilon:
            #print("   - explore")
            action_ix = random.randint(0, self.num_arms - 1)
                    
        # exploit best action
        else:
            #print("   - exploit")
            action_ix = np.argmax(self.estimated_reward)
            
        #print("   - action_ix: %d" % action_ix)
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
     
        self.estimated_reward[last_action_ix] =\
            self.estimated_reward[last_action_ix] + self.alpha * (reward_last_action + self.gamma * np.max(self.estimated_reward) - self.estimated_reward[last_action_ix])
        
#         print("   - cumulated_reward = ", self.cumulated_reward)
#         print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        self.update_epsilon(self.epsilon_original, self.iteration)
        
    def update_epsilon(self,epsilon_original, iteration):
        self.epsilon = epsilon_original / math.sqrt(iteration)
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        

# Thompson sampling - Beta distribution
class Thompson_sampling_mab:    
    
    def __init__(self, num_arms, distribution="beta", name="mab"):
        self.num_arms = num_arms
        self.name=name
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        self.distribution = distribution
        if distribution == "beta":
            self.alpha = [1] * num_arms
            self.beta = [1] * num_arms

            
    # pick action  
    def select_action(self):  
        
        #print("*** iteration #%d ***" % (self.iteration))
        
        # draw sample from beta distribution of each arm
        theta = [0] * self.num_arms
        for arm_ix in range(self.num_arms):
            theta[arm_ix] = np.random.beta(self.alpha[arm_ix], self.beta[arm_ix])
                
        # pick higher value of theta
        action_ix = np.argmax(theta)
        #print("   - action_ix: %d" % action_ix)
    
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        return action_ix
        
    # update rewards and params
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        # update params
        self.alpha[last_action_ix] = self.alpha[last_action_ix] + reward_last_action
        self.beta[last_action_ix] = self.beta[last_action_ix] + (1 - reward_last_action)
        
        # check safety
        if(self.alpha[last_action_ix] > 1):
            self.alpha[last_action_ix] = 1
        if(self.alpha[last_action_ix] <= 0):
            self.alpha[last_action_ix] = 0.001
        if(self.beta[last_action_ix] > 1):
            self.beta[last_action_ix] = 1
        if(self.beta[last_action_ix] <= 0):
            self.beta[last_action_ix] = 0.001
            
        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("*", self.name)
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
# Thompson sampling - Normal distribution
class Thompson_sampling_normal:    
    
    def __init__(self, num_arms, distribution="normal", name="mab"):
        self.num_arms = num_arms
        self.name=name
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        self.distribution = distribution
            
    # pick action  
    def select_action(self):  
        
        #print("*** iteration #%d ***" % (self.iteration))
        
        # draw sample from normal distribution of each arm
        theta = [0] * self.num_arms
        for arm_ix in range(self.num_arms):
            mean = self.estimated_reward[arm_ix]
            std = 1 / (self.count_action_selected[arm_ix] + 1)
            theta[arm_ix] = np.random.normal(mean, std)
                
        # pick higher value of theta
        action_ix = np.argmax(theta)
        #print("   - action_ix: %d" % action_ix)
    
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        return action_ix
        
    # update rewards and params
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] =\
            (self.estimated_reward[last_action_ix] * self.count_action_selected[last_action_ix] + reward_last_action) / (self.count_action_selected[last_action_ix] + 2)
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
            
        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("*", self.name)
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
# UCB1
class UCB1_mab:    
    
    def __init__(self, num_arms, name="mab"):
        self.num_arms = num_arms
        self.name=name
        self.estimated_reward = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.reward_history = np.array([])
        self.iteration = 1
                    
    # pick action  
    def select_action(self):  
        
        #print("*** iteration #%d ***" % (self.iteration))
        
        # draw sample from beta distribution of each arm
        ucb_value = [0] * self.num_arms
        for arm_ix in range(self.num_arms):
            ucb_value[arm_ix] = self.estimated_reward[arm_ix]\
            + math.sqrt((2*math.log(self.iteration)) / (self.count_action_selected[arm_ix] + 1))
                
        # pick higher value of ucb_value
        action_ix = np.argmax(ucb_value)
        #print("   - action_ix: %d" % action_ix)
    
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        return action_ix
        
    # update rewards and params
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("*", self.name)
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))


# Exp3
from numpy.random import choice
class Exp3_mab:    
    
    def __init__(self, gamma, num_arms, name="mab"):
        self.gamma = gamma
        self.name = name
        self.num_arms = num_arms
        self.estimated_reward = [-1] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        # Exp3 specific
        self.weights = [1] * num_arms
        self.probs = [1/num_arms] * num_arms
    
    # pick action  
    def select_action(self):  
        
        #print("*** iteration #%d ***" % (self.iteration))

        # draw sample from 'probs'
        norm_prob = self.probs / np.array(self.probs).sum()
        action_ix = choice(list(range(self.num_arms)), size=1, p=norm_prob, replace=False)
        # convert 1-element array to integer
        action_ix = action_ix.item() 
        
        #print("   - action_ix: %d" % action_ix)
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        return action_ix
        
    def update_reward(self,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        
        self.estimated_reward[last_action_ix] = reward_last_action/self.probs[last_action_ix]
        
        self.weights[last_action_ix] = self.weights[last_action_ix]\
        * math.exp((self.gamma * self.estimated_reward[last_action_ix]) / self.num_arms)
        
        self.probs[last_action_ix] = (1-self.gamma) * self.weights[last_action_ix] / sum(self.weights)\
        + self.gamma / self.num_arms
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))

        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        

# ---------------------------------------------------

# Q-learning
class Qlearning:    
    
    def __init__(self, epsilon_original, alpha, gamma, num_arms, num_states, name="ql"):
        
        self.epsilon_original = epsilon_original
        self.alpha = alpha
        self.gamma = gamma
        self.name = name
        self.epsilon = epsilon_original
        self.num_arms = num_arms
        self.num_sates = num_states
        
        # Initialize q-table values to 0
        self.Q = np.zeros((num_states, num_arms))
        
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self, state):  
               
        rand = random.uniform(0, 1)
                
        # explore
        if rand < self.epsilon:
            #print("   - explore")
            action_ix = random.randint(0, self.num_arms - 1)
                    
        # exploit best action
        else:
            #print("   - exploit")
            action_ix = np.argmax(self.Q[state])
            
        #print("   - action_ix: %d" % action_ix)
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,last_state, new_state, last_action_ix, reward_last_action):
        
#         print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
#         print("   - count_action_selected = ", self.count_action_selected)
        
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        
        self.estimated_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]
        
        self.Q[last_state][last_action_ix] =\
            self.Q[last_state][last_action_ix] + self.alpha \
            * (reward_last_action + self.gamma * np.max(self.Q[new_state]) - self.Q[last_state][last_action_ix])
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        #print("   - Q = ", self.Q)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        self.update_epsilon(self.epsilon_original, self.iteration)
        
    def update_epsilon(self,epsilon_original, iteration):
        self.epsilon = epsilon_original / math.sqrt(iteration)
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
        
# contextual exploration first
class Contextual_explfirst_mab:    
    
    def __init__(self, num_arms, name="mab"):
        
        self.satisfied_MAB = mab.Explfirst_mab(num_arms, name= name + "-satis")
        self.unsatisfied_MAB = mab.Explfirst_mab(num_arms, name= name + "-unsatis")
        
        self.name = name
        self.num_arms = num_arms
        self.estimated_reward = [-1] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self,context):  
        
        # print("*** iteration #%d ***" % (self.iteration))
        
        if context == 0:
            action_ix = self.unsatisfied_MAB.select_action()
        else:
            action_ix = self.satisfied_MAB.select_action()
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,context,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        if context == 0:
            self.unsatisfied_MAB.update_reward(last_action_ix, reward_last_action)
        else:
            self.satisfied_MAB.update_reward(last_action_ix, reward_last_action)
            
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] = reward_last_action
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
        
# ---------
# contextual exploration first
class Contextual_egreedy_mab:    
    
    def __init__(self, epsilon_original, num_arms, name="mab"):
        
        self.satisfied_MAB = mab.Epsilongreedy_mab(epsilon_original, num_arms, name= name + "-satis")
        self.unsatisfied_MAB = mab.Epsilongreedy_mab(epsilon_original, num_arms, name= name + "-unsatis")
        
        self.epsilon_original = epsilon_original
        self.name = name
        self.epsilon = epsilon_original
        self.num_arms = num_arms
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self,context):  
        
        # print("*** iteration #%d ***" % (self.iteration))
        
        if context == 0:
            action_ix = self.unsatisfied_MAB.select_action()
        else:
            action_ix = self.satisfied_MAB.select_action()
        
        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,context,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        if context == 0:
            self.unsatisfied_MAB.update_reward(last_action_ix, reward_last_action)
        else:
            self.satisfied_MAB.update_reward(last_action_ix, reward_last_action)
            
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] = reward_last_action
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))
        
        
# ---------
# contextual exploration first
class Contextual_egreedy_24_mab:    
    
    def __init__(self, epsilon_original, num_arms, name="mab"):
        
        NUM_ARMS = 24
        
        self.MAB_per_context = []
        
        for context_ix in range(NUM_ARMS):
            self.MAB_per_context.append(mab.Epsilongreedy_mab(epsilon_original, num_arms, name=context_ix))
        
        self.epsilon_original = epsilon_original
        self.name = name
        self.epsilon = epsilon_original
        self.num_arms = num_arms
        self.estimated_reward = [0] * num_arms
        self.count_action_selected = [0] * num_arms
        self.cumulated_reward = [0] * num_arms
        self.mean_reward = [0] * num_arms
        self.iteration = 1
        self.reward_history = np.array([])
        
        self.current_action_ix = -1
        self.num_action_switch = 0
    
    # pick action  
    def select_action(self,context):  
        
        # print("*** iteration #%d ***" % (self.iteration))
        
        action_ix = self.MAB_per_context[context].select_action()

        self.count_action_selected[action_ix] = self.count_action_selected[action_ix] + 1
        
        if not self.current_action_ix == action_ix:
            self.num_action_switch = self.num_action_switch + 1
        
        self.current_action_ix = action_ix
        
        return action_ix
        
    def update_reward(self,context,last_action_ix, reward_last_action):
        
        #print("last_action_ix: %d, reward_last_action = %.2f" % (last_action_ix,reward_last_action))
        #print("   - count_action_selected = ", self.count_action_selected)
        
        self.MAB_per_context[context].update_reward(last_action_ix, reward_last_action)
            
        self.cumulated_reward[last_action_ix] = self.cumulated_reward[last_action_ix] + reward_last_action
        self.estimated_reward[last_action_ix] = reward_last_action
        
        #print("   - cumulated_reward = ", self.cumulated_reward)
        #print("   - estimated_reward = ", self.estimated_reward)
        
        self.reward_history = np.append(self.reward_history, np.array(reward_last_action))
        
        self.mean_reward[last_action_ix] =\
            self.cumulated_reward[last_action_ix] / self.count_action_selected[last_action_ix]

        self.iteration = self.iteration + 1
        
    def print_summary(self):
        print("   - count_action_selected = ", self.count_action_selected)
        print("   - cumulated_reward = ", self.cumulated_reward)
        print("   - mean_reward = ", self.mean_reward)
        print("   - estimated_reward = ", self.estimated_reward)
        print("   - best_action = ",  np.argmax(self.estimated_reward))