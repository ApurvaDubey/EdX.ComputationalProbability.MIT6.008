#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

# generate emission and transition matrix, and mapping dictionaries
if 1 == 1:
    # Create dictionary mapping of matrix index to keys
    hs_idx = {}
    idx_hs = {}
    for (idx,hs) in enumerate(all_possible_hidden_states):
        hs_idx[hs] = idx
        idx_hs[idx] = hs
        if idx == 0:
            t = transition_model(hs)
       
    os_idx = {}
    idx_os = {}
    for (idx,os) in enumerate(all_possible_observed_states):
        os_idx[os] = idx
        idx_os[idx] = os
    #-----------------------------------------------------

    #-----------------------------------------------------
    # Transition pobabilities
    transition = np.array(np.zeros((440,440)))

    for (idx,hs) in enumerate(all_possible_hidden_states):
       
        # row number to be updated is
        r = hs_idx[hs]
       
        # columns to be updated
        x = transition_model(hs)
        for k in x.keys():
            v = x[k]
            c = hs_idx[k]
            transition[r,c] = v   
    #-----------------------------------------------------

    #-----------------------------------------------------
    # Emission probabilities
    emission = np.array(np.zeros((440,96)))

    for (idx,hs) in enumerate(all_possible_hidden_states):
       
        # row number to be updated is
        r = hs_idx[hs]

        # columns to be updated
        x = observation_model(hs)
        for k in x.keys():
            v = x[k]
            c = os_idx[k]
            emission[r,c] = v
    #-----------------------------------------------------
    
    #-----------------------------------------------------
    # Prior Distribution
    prior = [1.0/96 if k[2] == "stay" else 0 for k in all_possible_hidden_states]
    #print "prior..."
    #print np.shape(prior)
    #-----------------------------------------------------

    
# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """
    #observations = ([(6, 0), (6, 2), (7, 2), (7, 3), (7, 4), (7, 5), (6, 5), (5, 6), None, (7, 7)])
    num_time_steps = len(observations)
  

    ####################################################################
    # Inputs
    #-----------------------------------------------------



    # Observations
    observations = observations
    #-----------------------------------------------------

    ####################################################################
    # parameters for forward/backward algorithm
    alpha = np.zeros(shape=(num_time_steps,440)) # timesteps x hidden states
    beta = np.zeros(shape=(num_time_steps,440)) # timesteps x hidden states

    #print "Initial value of alpha and beta"
    #print alpha, '\n', beta

    # initialize alpha and beta
    alpha[0] = prior

    beta[len(observations)-1] = 1.0/len(all_possible_hidden_states)
    ####################################################################

    ####################################################################
    #print "Running forward algorithm"

    # Go forward
    for i in range(len(observations)-1):

        if observations[i] == None:
            alpha[i+1] = np.matrix(np.array(alpha[i]) * np.matrix(transition)) # prior message x transition probability                             

        else:
            alpha[i+1] = np.matrix(np.array(alpha[i]) * np.array(emission[:,os_idx[observations[i]]])) * np.matrix(transition) 
                         # prior message x emision probability x transition probability
                    
        alpha[i+1] = alpha[i+1]/sum(list(alpha[i+1])) # normalize

        #print "Iteration : " + str(i) + "\n" + str(alpha)

    #print "Final alpha is: \n -------------------------"
    #print len(np.matrix(np.array(alpha[0]) * np.array(emission[:,os_idx[observations[0]]])))
    #print alpha

    # Go backward
    #print "Running backward algorithm"
    for i in range(len(observations)-2,-1,-1):

        if observations[i+1] == None:
            beta[i] = np.matrix(np.array(beta[i+1])) * np.transpose(np.matrix(transition)) # prior message x transition probability

        else: 
            beta[i] = np.matrix(np.array(beta[i+1]) * np.array(emission[:,os_idx[observations[i+1]]])) * np.transpose(np.matrix(transition))
                    # prior message x emision probability x transition probability

        beta[i] = beta[i]/sum(list(beta[i])) # normalize                
        
    #print "Final beta is: \n -------------------------"
    #print beta
    ####################################################################
        
    ####################################################################
    #print "Calculating marginals"

    marginals = [None] * num_time_steps        

    for i in range(0,num_time_steps):
       
        x = robot.Distribution()

        if observations[i] == None:
            temp = (alpha[i]) * (beta[i])  # marginal = alpha x beta
            
        else:
            temp = (alpha[i]) * (beta[i]) * np.array(emission[:,os_idx[observations[i]]]) # marginal = alpha x beta x emission probability
        
        for j in range(0,440):
            if temp[j] > 0:
                x[idx_hs[j]] = temp[j]
               
        x.renormalize()
       
        marginals[i] = x

    #print "\nmarginals: \n" + str(marginals)
    ####################################################################    
    return marginals


def Viterbi(observations):
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
    #observations = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    fm = np.zeros(shape=(num_time_steps-1,440))
    tb = np.zeros(shape=(num_time_steps-1,440))

    # forward pass
    for i in range(len(observations)-1):
        # print i
        e = np.array(emission[:,os_idx[observations[i]]])

        x_fm = []
        x_tb = []

        if i==0:
            for j in range(0,440):
                em = np.array([-careful_log(p) -careful_log(prior[w]) for w,p in enumerate(e)]) # emission
                tr = np.array([-careful_log(p) for p in transition[:,j]]) # transition
                em_tr = em+tr
 
                x_fm.append(min(em_tr))
                x_tb.append(list(em_tr).index(min(em_tr)))
            
        if i > 0:
            for j in range(0,440):
                em = np.array([-careful_log(p) for p in e]) # emission
                tr = np.array([-careful_log(p) for p in transition[:,j]]) # transition
                em_tr = em+tr+fm[i-1]
 
                x_fm.append(min(em_tr))
                x_tb.append(list(em_tr).index(min(em_tr)))

        fm[i] = x_fm
        tb[i] = x_tb

    # backward pass / traceback call
    tb_final = []

    # last estimated state
    e = np.array(emission[:,os_idx[observations[-1]]])
    em = np.array([-careful_log(p) for p in e]) # emission

    # last state before going in the reverse direction
    em_fm = em+fm[-1]

    #print "last state..."
    #print list(em_fm).index(min(em_fm))
       
    tb_final.append(list(em_fm).index(min(em_fm))) # last state

        
    #print "-------------"
    for h in reversed(range(len(observations)-1)):
        tb_final.insert(0,int(tb[h][tb_final[0]]))
        
    #print tb_final

    estimated_hidden_states = [idx_hs[t] for t in tb_final]

    #print tb_final
    #print estimated_hidden_states

    #print fm[0]

    return estimated_hidden_states


def second_best(observations):
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


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
##    if use_graphics:
##        app = graphics.playback_positions(hidden_states,
##                                          observations,
##                                          estimated_states,
##                                          marginals)
##        app.mainloop()


if __name__ == '__main__':
    main()
