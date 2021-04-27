# Making the imports needed
import numpy as np
import random
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
import math
import statistics as stats

# Defining a function that generates a list of random uniform numbers, with length specified by the user - testing purposes
def random_list(n):
    sequence = []
    for i in range(n):
        number = np.random.uniform(0,1)
        sequence.append(number)
    return sequence  

# Defining a function that randomly mixes 2 lists, equal proportions - testing purposes
def equal_mix(list1, list2, n):
    list3 = []
    for i in range(int(n/2)):
        selection1 = np.random.choice(list1)
        list1.remove(selection1)
        list3.append(selection1)
        selection2 = np.random.choice(list2)
        list2.remove(selection2)
        list3.append(selection2)
    return list3

# Defining a function that randomly mixes 2 lists based on a single biased coin - testing purposes 
def unequal_mix_1b_coin(list1, list2, n, p1):
    list3 = []
    for i in range(n):
        if np.random.uniform(0,1) > p1:
            selection1 = np.random.choice(list1)
            list1.remove(selection1)
            list3.append(selection1)
        else:
            selection2 = np.random.choice(list2)
            list2.remove(selection2)
            list3.append(selection2)
    return list3
    
# Defining a function that mixes 2 lists based on 2 biased coins, different from Allison Mix -> starting from the first list - testing purposes
def random_mix(list1, list2, endpoint, alpha1 = 0.5, alpha2 = 0.5): # max recommended endpoint = (len(list)-1)
    list3 = []
    selection = np.random.choice(list1)
    list1.remove(selection)
    list3.append(selection)
    
    list1_selections = 1
    list2_selections = 0
    
    state1 = True
    #state2 = False
    
    i = 1 # start from 1 because we've already chose an element from list1
    
    while i < endpoint:
        if state1:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= alpha1:
                # Randomly choose a number from list 2
                selection = np.random.choice(list2)
                list2.remove(selection)
                list3.append(selection)
                list2_selections += 1
                state1 = False
                #state2 = True
            else:
                # Randomly choose a number from list 1
                selection = np.random.choice(list1)
                list1.remove(selection)
                list3.append(selection)
                list1_selections += 1
                state1 = True
                #state2 = False
                
        else:
                # Flip the coin and make decision
                prob = np.random.uniform(0,1)
                if prob <= alpha2:
                    # Randomly choose a number from list 1
                    selection = np.random.choice(list1)
                    list1.remove(selection)
                    list3.append(selection)
                    list1_selections += 1
                    state1 = True
                    #state2 = False
                else:
                    # Randomly choose a number from list 2
                    selection = np.random.choice(list2)
                    list2.remove(selection)
                    list3.append(selection)
                    list2_selections += 1
                    state1 = False
                    #state2 = True
        i += 1

    return list3, list1_selections, list2_selections
    
# Testing random_mix function
um_l1 = np.random.normal(0.35, 0.1, 100)
um_l2 = np.random.uniform(0, 1, 100)
um_l3, um_l1_selections, um_l2_selections = unequal_mix(list(um_l1), list(um_l2), endpoint=100, alpha1=0.9, alpha2=0.9)

print('Au fost selectate {} valori din lista 1'.format(um_l1_selections))
print('Au fost selectate {} valori din lista 2'.format(um_l2_selections))

# Defining a function that mixes 2 lists based on 2 biased coins in order based on lists's index - ALLISON MIX
def allison_mix(list1, list2, endpoint, startpoint = 1, alpha1 = 0.5, alpha2 = 0.5): # max endpoint = (len(list1)-1)
    startpoint -= 1 # minus 1 because indexing in python starts from 0
    
    list3 = []
    list3.append(list1[startpoint])
    
    list1_selections = 1 # list storing the number of selections made from the first list 
    list2_selections = 0 # list storing the number of selections made from the second list
    
    # We are in the first list chosen as input
    state1 = True
    #state2 = False
    
    while startpoint < (endpoint-1): #endpoint - 1 because indexing in python starts from 0
        if state1:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= alpha1:
                # Switch and get the value from list2
                list3.append(list2[startpoint+1])
                list2_selections += 1
                state1 = False
            else:
                # Remain in list1
                list3.append(list1[startpoint+1])
                list1_selections += 1
                state1 = True
                
        else:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= alpha2:
                # Switch and get the value from list1
                list3.append(list1[startpoint+1])
                list1_selections += 1
                state1 = True
            else:
                # Remain in list2
                list3.append(list2[startpoint+1])
                list2_selections += 1
                state1 = False
        startpoint += 1
                
    return list3, list1_selections, list2_selections
    
# Testing allison_mix function
am_l1 = np.random.normal(0.35, 0.1, 5)
am_l2 = np.random.uniform(0, 1, 5)
am_l3, am_l1_selections, am_l2_selections = allison_mix(list(am_l1), list(am_l2), startpoint = 4,endpoint=5, alpha1=0.58, alpha2=0.53)

print('Au fost selectate {} valori din lista 1'.format(am_l1_selections))
print('Au fost selectate {} valori din lista 2'.format(am_l2_selections))

# Test lists to show how the Allison Mix is created
ones = np.random.normal(2, 0.2, 50)
twos = np.random.normal(8, 0.6, 50)

am_example, no_list1, no_list2 = allison_mix(ones, twos, endpoint = 50, alpha1=0.4, alpha2=0.7)

#Plotting the autocorrelations
%matplotlib inline

plt.plot(am_example)
plt.show()

# Battle of the sexes - game set up
def bos_game(woman, n, p1 = 0.6, p2 = 0.6):
    '''
    Function that generates the results for the game Battle of Sexes through Monte Carlo simulations
    
    woman: boolean, should take values True or False
    
    n: number of simulations
    
    p1: the probability of choosing theater as a woman
    p2: the probability of choosing a football match as a man
    
    Function returns lists with the strategies for women and mans. 1=Theater, 0=Football Match
    '''
    #Initialize the list with the strategies
    results = []
    
    i = 0 # start the MC simulations from 0
    
    # Simulates over a numbers of samples specified by user
    while i < n:
        # Results for woman
        if woman:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= p1:
                results.append(1) # adauga 1 in lista cu strategii daca p generata este mai mica decat cea specificata
            else:
                results.append(0) # adauga 0 in lista cu strategii daca p generata este mai mare decat cea specificata
        
        # Results for man
        else:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= p2:
                results.append(0) # adauga 0 in lista intrucat 0 este strategia preferata de barbati
            else:
                results.append(1) # adauga 1 in lista          
        
        i += 1
        
    return results
    
# Formula for the autocorrelation of an Allison Mixture -> (Gunn, Allison, & Abbott, 2014)
def am_autocorr(list1, list2, list3, alpha1, alpha2):
    miu1 = np.mean(list1)
    miu2 = np.mean(list2)
    var = np.var(list3)
    
    autocorr = ((miu1-miu2)**2) * ((alpha1*alpha2)/((alpha1+alpha2)**2)) * (1-alpha1-alpha2) / var
    
    return autocorr
    
# Formula for the autocovariance of an Allison Mixture ->(Gunn, Allison, & Abbott, 2014)
def am_autocov(list1, list2, alpha1, alpha2):
    miu1 = np.mean(list1)
    miu2 = np.mean(list2)
    
    covar = ((miu1-miu2)**2) * ((alpha1*alpha2)/((alpha1+alpha2)**2)) * (1-alpha1-alpha2)

    return covar
    
# Simulate 10.000 scenarios of BoS and get the strategies for woman and man

woman_strategy = bos_game(True, 10000, p1=0.6)
man_strategy = bos_game(False, 10000, p2=0.6)

# Form the Allison Mix
alpha1 = 0.42
alpha2 = 0.54

am_bos, no_woman, no_man = allison_mix(woman_strategy, man_strategy, endpoint = 10000, alpha1 = alpha1, alpha2 = alpha2)

print('The number of samples from woman selected is {} and from man is {}'.format(no_woman, no_man))

# covariance through classic method
w_strategy = pd.Series(woman_strategy)
m_strategy = pd.Series(man_strategy)

print('Covarianta conform formulei din statistica este: {}'.format(w_strategy.cov(m_strategy)))

# correlation through classic method
print('Corelatia conform formulei din statistica este: {}'.format(w_strategy.corr(m_strategy, method ='pearson')))

# covariance through literature method
covarianta = am_autocov(woman_strategy, man_strategy, alpha1 = 0.42, alpha2 = 0.54)
print('Autocovarianta Allison Mixture din Battle of Sexes este: {}'.format(covarianta))

# correlation through literature method
corelatia = am_autocorr(woman_strategy, man_strategy, am_bos, alpha1 = 0.42, alpha2 = 0.54) 
print("Autocorelatia Allison Mixture din Battle of Sexes este: {}".format(corelatia))

# Some extra imports to help us identify the right alpha1, alpha2
from IPython.display import display
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

alpha1 = widgets.BoundedFloatText(
    value=0.50,
    min=0,
    max=1,
    step=0.01,
    description='alpha1:',
    disabled=False)

alpha2 = widgets.BoundedFloatText(
    value=0.60,
    min=0,
    max=1,
    step=0.01,
    description='alpha2:',
    disabled=False)

interact(am_autocov, list1=fixed(woman_strategy), list2=fixed(man_strategy), alpha1=alpha1, alpha2=alpha2)
print('Covarianta teoretica este: {}'.format(w_strategy.cov(m_strategy)))

# Check again the covariance
covarianta = am_autocov(woman_strategy, man_strategy, alpha1 = alpha1, alpha2 = alpha2)
print('Autocovarianta Allison Mixture din Battle of Sexes este: {}'.format(covarianta))

# Check again the correlation
corelatia = am_autocorr(woman_strategy, man_strategy, am_bos, alpha1 = alpha1, alpha2 = alpha2) 
print("Autocorelatia Allison Mixture din Battle of Sexes este: {}".format(corelatia))

# A DIFFERENT MIX - testing purposes
alpha1 = 0.42
alpha2 = 0.21
am_bos, no_woman, no_man = random_mix(woman_strategy, man_strategy, endpoint = 10000, alpha1 = alpha1, alpha2 = alpha2)
print('The number of samples from woman selected is {} and from man is {}'.format(no_woman, no_man))

# Adding the lists with the strategies to a data frame and get them ready to export
df = pd.DataFrame(data={"Woman Strategy": woman_strategy, "Man Strategy": man_strategy, "Allison Mix": am_bos})
df.to_csv("./file.csv", sep=',',index=False)

# Compute the covariance for all possible alpha1, alpha2 and export the data
list_alpha1 = []
list_alpha2 = []
output = []

for alpha1 in np.arange(0.01,1,0.01):
    for alpha2 in np.arange(0.01,1,0.01):
        list_alpha1.append(alpha1)
        list_alpha2.append(alpha2) 
        output.append(am_autocov(woman_strategy, man_strategy, alpha1 = alpha1, alpha2 = alpha2))

data = pd.DataFrame(data={"alpha1": list_alpha1, "alpha2": list_alpha2, "am_covariance": output})
data.to_csv("./all_10000_sim.csv", sep=',',index=False)

# Split or Steal - game set up
def split_or_steal(player1, n, p1 = 0.6, p2 = 0.6):
    '''
    Function that generates the results for the game Split or Steal through Monte Carlo simulations
    
    player1: boolean, should take value True or False
    
    n: number of simulations
    
    p1: the probability of choosing strategy Split for player1
    p2: the probability of choosing strategy Split for player2
    
    Function returns lists with the strategies for player1 and player2. 1=Split, 0=Steal
    '''
    #Initialize the list with the strategies
    results = []
    
    i = 0 # start the MC simulations from 0
    
    # Simulates over a numbers of samples specified by user
    while i < n:
        if player1:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= p1:
                results.append(1) # adauga 1 in lista cu strategii daca p generata este mai mica decat cea specificata
            else:
                results.append(0) # adauga 0 in lista cu strategii daca p generata este mai mare decat cea specificata
        
        else:
            # Flip the coin and make decision
            prob = np.random.uniform(0,1)
            if prob <= p2:
                results.append(1) # adauga 1 in lista intrucat 0 este strategia pentru player2
            else:
                results.append(0) # adauga 0 in lista          
        
        i += 1
        
    return results
    
# Get the list with the strategies for both players
player1_strategies = split_or_steal(player1=True, n=10000, p1=0.6667)
player2_strategies = split_or_steal(player1=False, n=10000, p2=0.3333)

# The covariance and correlation were calculated based on the same steps as for Battle of the Sexes so the code was not included again as it is the same
