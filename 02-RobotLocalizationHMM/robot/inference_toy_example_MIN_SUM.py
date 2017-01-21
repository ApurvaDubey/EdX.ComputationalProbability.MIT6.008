#!/usr/bin/env python

import collections
import sys
import numpy as np

# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
    
def Viterbi():
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    observations = ['H','H','T','T','T'] # observed states
    transition = np.array([[0.75,0.25],[0.25,0.75]]) # transition probabilities
    emission = np.array([[0.5,0.5],[0.25,0.75]]) # emission probabilities

    prior = np.array([0.5,0.5]) # priors
    
    os_idx = {'H':0,'T':1} # indices for mapping observed states to index
    idx_hs = {0:'fair',1:'biased'} # indices for mapping index states to hidden state
    
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    fm = np.zeros(shape=(num_time_steps-1,2)) # forward messages
    tb = np.zeros(shape=(num_time_steps-1,2)) # tracebacks

    # forward pass
    for i in range(len(observations)-1):
 
        x_given_y = np.array(emission[:,os_idx[observations[i]]]) # get distribution of hidden state given the observed state

        tmp_fm = [] # temporary array for messages
        tmp_tb = [] # temporary array for tracebacks

        for j in range(0,2):
            if i==0: # prior will be taken into account only for the first observation
                log_x_given_y = np.array([-careful_log(x) -careful_log(prior[k]) for k,x in enumerate(x_given_y)]) # get log
            else:
                log_x_given_y = np.array([-careful_log(x) for x in x_given_y]) # get log
            tr = np.array([-careful_log(t) for t in transition[:,j]]) # transition
            log_x_given_y_tr = log_x_given_y+tr+fm[i-1]
 
            tmp_fm.append(min(log_x_given_y_tr))
            tmp_tb.append(list(log_x_given_y_tr).index(min(log_x_given_y_tr)))

        fm[i] = tmp_fm
        tb[i] = tmp_tb

    # backward pass / traceback calls
    tb_final = []

    # last estimated state
    x_given_y = np.array(emission[:,os_idx[observations[-1]]]) # get distribution of hidden state for the last observed state
    em = np.array([-careful_log(p) for p in x_given_y]) # 

    # last state before going in the reverse direction     
    tb_final.append(list(em+fm[-1]).index(min(em+fm[-1]))) # last state

    for h in reversed(range(len(observations)-1)):
        tb_final.insert(0,int(tb[h][tb_final[0]]))
        
    estimated_hidden_states = [idx_hs[t] for t in tb_final]

    print "messages..."
    print fm

    print "\nestimates hidden states..."    
    print (estimated_hidden_states)
    
    return estimated_hidden_states

# -----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    print('Running Viterbi...')
    estimated_states = Viterbi()
    print("\n")
    
