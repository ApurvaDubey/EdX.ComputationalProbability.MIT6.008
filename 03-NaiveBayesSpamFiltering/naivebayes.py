import sys
import os.path
import numpy as np
import math

import util

globvar_spam_cnt = 0
globvar_ham_cnt = 0

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here

    d = {}

    for i in range(len(file_list)):
        for w in util.get_words_in_file(file_list[i]):
            d[w] = d.get(w,set([])) | set([i])

    d_final = {k:len(d[k]) for k in d.keys()}
    #print [(k,d_final[k]) for k in d_final.keys()][0:10]

    print len(d_final)
    return d_final

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here

    lenght_of_list = len(file_list)
    #print lenght_of_list
        
    d = get_counts(file_list)

    d_final_log = {k:math.log((1.0*(d[k]+1.0)/(lenght_of_list+2.0))) for k in d.keys()}

    #print [(k,d_final_log[k]) for k in d_final_log.keys()][0:10]
    
    return d_final_log
    

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here

    log_probabilities_by_category = [get_log_probabilities(file_lists_by_category[0]),get_log_probabilities(file_lists_by_category[1])]

    #-----------------------
    
    num_spams = len(file_lists_by_category[0])
    num_hams = len(file_lists_by_category[1])
    tot_emails = num_spams + num_hams
    
    global globvar_spam_cnt
    globvar_spam_cnt = num_spams
    global globvar_ham_cnt
    globvar_ham_cnt = num_hams

    log_prior_by_category = [math.log(1.0*num_spams/tot_emails),math.log(1.0*num_hams/tot_emails)]

    #log_prior_by_category = [math.log(0.001),math.log(0.999)]
    #You correctly classified 30 out of 49 spam emails, and 51 out of 51 ham emails.
   
    #log_prior_by_category = [math.log(0.1),math.log(0.9)]
    #You correctly classified 36 out of 49 spam emails, and 49 out of 51 ham emails.

    #log_prior_by_category = [math.log(0.5),math.log(0.5)]
    #You correctly classified 38 out of 49 spam emails, and 48 out of 51 ham emails.

    #log_prior_by_category = [math.log(0.9),math.log(0.1)]
    #You correctly classified 40 out of 49 spam emails, and 48 out of 51 ham emails.

    #log_prior_by_category = [math.log(0.999),math.log(0.001)]
    #You correctly classified 41 out of 49 spam emails, and 46 out of 51 ham emails
  
    return log_probabilities_by_category, log_prior_by_category

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here

    p_spam = log_prior_by_category[0]
    p_ham = log_prior_by_category[1]

    #print p_spam, p_ham
    
    for w in util.get_words_in_file(email_filename):
        #print len(file_lists[0]), len(file_lists[1])
        #p_spam = p_spam + log_probabilities_by_category[0].get(w,0)
        #p_ham = p_ham + log_probabilities_by_category[1].get(w,0)
        p_spam = p_spam + log_probabilities_by_category[0].get(w,math.log(1.0/(globvar_spam_cnt+2.0)))
        p_ham = p_ham + log_probabilities_by_category[1].get(w,math.log(1.0/(globvar_ham_cnt+2.0)))
        
    if p_spam > p_ham:
        category = 'spam'
    else:
        category = 'ham'
    
    return category

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    
    ### Read arguments
    if 1>2: 
    #if len(sys.argv) != 4:
        print (USAGE % sys.argv[0])
    
    os.chdir('C:/Users/Tulika/Documents/01-DataScience/ComputationalProb/CodingProjects/03-NaiveBayesSpamFiltering/naivebayes/data')
    #testing_folder = sys.argv[1]
    #(spam_folder, ham_folder) = sys.argv[2:4]
    testing_folder = "testing"
    spam_folder = "spam"
    ham_folder = "ham"
    ### Learn the distributions
    
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    
    (log_probabilities_by_category, log_priors_by_category) = learn_distributions(file_lists)

    #print "part a completed"
    
    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
