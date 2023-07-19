#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[22]:


#Policy evaluation
grid_height = 7
grid_width = grid_height
action = 4
reward = -1
dis = 1
Break = True

post_value_table = np.array([[-1, -1, -1, -1, -1, -1, -1], 
                             [-1, -1, -1, -1, -1, -1, -1], 
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, 0]], dtype = float)
    
for iteration in range(100):
    next_value_table = np.array([[-1, -1, -1, -1, -1, -1, -1], 
                                 [-1, -1, -1, -1, -1, -1, -1], 
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, 0]], dtype = float)
    # print result
    print('Iteration: {} \n{}\n'.format(iteration, post_value_table))
        
    for i in range(grid_height):
        for j in range(grid_width):
                value_t = 0
                if i == 6 and j == 6: # 종점 update X
                    Break = False
                    break
                for act in range(action):  #0, 1, 2, 3 (상하좌우)
                        
                        if act == 0:    #Up
                            if i != 0:
                                i_ = i-1
                                j_ = j
                            else :
                                i_ = i
                                j_ = j
                        elif act == 1: #Down
                            if i != 6:
                                i_ = i+1
                                j_ = j
                            else :
                                i_ = i
                                j_ = j
                        elif act == 2: #Left
                            if j != 0:
                                i_ = i
                                j_ = j-1
                            else:
                                i_ = i
                                j_ = j
                        elif act == 3: #Right
                            if j != 6:
                                i_ = i
                                j_ = j+1
                            else:
                                i_ = i
                                j_ = j
                        
                        
                        if i == 0 and j == 2:                                   # 해당 state의 함정 reward: -100
                            value = 0.25 * (-100 + dis*post_value_table[i_][j_])
                        elif i == 1 and j == 2:                              # 함정
                            value = 0.25 * (-100 + dis*post_value_table[i_][j_])
                        elif i == 3 and ((j == 4) or (j == 5)):              # 함정
                            value = 0.25 * (-100 + dis*post_value_table[i_][j_])
                        elif i == 6 and ((j == 2) or (j == 3)):              # 함정
                            value = 0.25 * (-100 + dis*post_value_table[i_][j_])
                        else :
                            value = 0.25 * (reward+ dis*post_value_table[i_][j_])
                        value_t += value
                        
                next_value_table[i][j] = round(value_t, 3) #반올림
        if Break == False:  # 종점 update X
            Break = True
            break
    iteration += 1
    post_value_table = next_value_table
                


# In[31]:


#Policy Improvement
def policy_improvement(value, action, policy, grid_width = 7):
    
    grid_height = grid_width
    
    action_match = ['Up', 'Down', 'Left', 'Right']
    action_table = []
    
    # get Q-func.
    for i in range(grid_height):
        for j in range(grid_width):
            q_func_list=[]
            if i==j and i==6:   #종점
                action_table.append('T')
            else:
                for k in range(len(action)):
                    if k == 0:   #Up
                        if i != 0:
                            i_ = i-1
                            j_ = j
                        else :
                            i_ = i
                            j_ = j
                    elif k == 1: #Down
                        if i != 6:
                            i_ = i+1
                            j_ = j
                        else :
                            i_ = i
                            j_ = j
                    elif k == 2: #Left
                        if j != 0:
                            i_ = i
                            j_ = j-1
                        else:
                            i_ = i
                            j_ = j
                    elif k == 3: #Right
                        if j != 6:
                            i_ = i
                            j_ = j+1
                        else:
                            i_ = i
                            j_ = j
                                
                    q_func_list.append(post_value_table[i_][j_])
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)] 

                # update policy
                policy[i][j]= [0]*len(action)     # initialize q-func_list
                for y in max_actions :
                    policy[i][j][y] = (1 / len(max_actions))

                # get action
                idx = np.argmax(policy[i][j])
                action_table.append(action_match[idx])
    action_table=np.asarray(action_table).reshape((grid_height, grid_width))                
    
    print('Updated policy is :\n{}\n'.format(policy))
    print('at each state, chosen action is :\n{}'.format(action_table))
    
    return policy


# In[32]:


grid_width = 7
grid_height = grid_width
action = [0, 1, 2, 3]  # up, down, left, right
policy = np.empty([grid_height, grid_width, len(action)], dtype=float)
for i in range(grid_height):
    for j in range(grid_width):
        for k in range(len(action)):
            if i==j and i == 6:
                policy[i][j]=0.00
            else :
                policy[i][j]=0.25
policy[0][0] = [0] * grid_width
policy[6][6] = [0] * grid_width


# In[33]:


updated_policy = policy_improvement(value, action, policy)


# In[34]:


#Value Iteration
grid_height = 7
grid_width = grid_height
action = 4
reward = -1
dis = 1
Break = True

post_value_table = np.array([[-1, -1, -1, -1, -1, -1, -1], 
                             [-1, -1, -1, -1, -1, -1, -1], 
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, -1],
                             [-1, -1, -1, -1, -1, -1, 0]], dtype = float)
    
for iteration in range(100):
    next_value_table = np.array([[-1, -1, -1, -1, -1, -1, -1], 
                                 [-1, -1, -1, -1, -1, -1, -1], 
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, 0]], dtype = float)
    # print result
    print('Iteration: {} \n{}\n'.format(iteration, post_value_table))
        
    for i in range(grid_height):
        for j in range(grid_width):
                value_t = 0
                if i == 6 and j == 6: # 종점 update X
                    Break = False
                    break
                value_t_list= []
                for act in range(action):  #0, 1, 2, 3 (상하좌우)
                        
                        if act == 0:    #Up
                            if i != 0:
                                i_ = i-1
                                j_ = j
                            else :
                                i_ = i
                                j_ = j
                        elif act == 1: #Down
                            if i != 6:
                                i_ = i+1
                                j_ = j
                            else :
                                i_ = i
                                j_ = j
                        elif act == 2: #Left
                            if j != 0:
                                i_ = i
                                j_ = j-1
                            else:
                                i_ = i
                                j_ = j
                        elif act == 3: #Right
                            if j != 6:
                                i_ = i
                                j_ = j+1
                            else:
                                i_ = i
                                j_ = j
                        
                        
                        if i == 0 and j == 2:                                   # 해당 state의 함정 reward: -100
                            value = (-100 + dis*post_value_table[i_][j_])
                        elif i == 1 and j == 2:                              # 함정
                            value = (-100 + dis*post_value_table[i_][j_])
                        elif i == 3 and ((j == 4) or (j == 5)):              # 함정
                            value = (-100 + dis*post_value_table[i_][j_])
                        elif i == 6 and ((j == 2) or (j == 3)):              # 함정
                            value = (-100 + dis*post_value_table[i_][j_])
                        else :
                            value = (reward+ dis*post_value_table[i_][j_])
                        value_t_list.append(value)
                        
                next_value_table[i][j] = max(value_t_list)
        if Break == False:  # 종점 update X
            Break = True
            break
    iteration += 1
    post_value_table = next_value_table


# In[ ]:




