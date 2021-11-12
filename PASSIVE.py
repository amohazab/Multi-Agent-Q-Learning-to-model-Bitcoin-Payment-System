import numpy as np
import pandas as pd
import openpyxl
import random
import sys
import json
from matplotlib import pylab as plt


BS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]



# Define parameters:
gamma =  0.3
alpha = 0.1



bidders_mv = [71,7, 56, 88,4,1,16, 25, 32, 39, 12, 48, 10,4, 3,40,14,4,12,30]
#bidders_mv = [18 , 16 , 12 ,11 , 10, 6, 3, 2, 2, 1]
n_bidders =len(bidders_mv)

max_rev = np.sum(bidders_mv)
n_iter =800000 #1,000,000

n_actions = np.max(bidders_mv)

#constant_supply = 4

        

# Define q-value table for the users/bidders

qval_bidders = [np.zeros([int(bidders_mv[i])+1 , int(np.max(bidders_mv))+1]) for i in range(0,n_bidders)]

# qval_bidders: actions , previous market clearing price


# marginal values of the bidders:


bids = np.zeros(n_bidders)

def softmax(av, epsilon):
    av_ep = av / epsilon
    valid = False
    while (not valid):
        if (np.max(np.exp(av_ep)) > 2**100):
            av_ep = av_ep/10
        else:
            valid = True
        
    softm = ( np.exp(av_ep) / np.sum( np.exp(av_ep)))
    return softm




# define the take_action function for a specific bidder:
def take_action(b , pc): #b: bidder index/id AND state: market clearing price
    p=softmax(qval_bidders[b][: , int(pc)] , epsilon )
    action = np.random.choice( list(range(0 , bidders_mv[b]+1)) , p=p)
    return int(action)



def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w


def get_best_action(v):
    sv=sorted(v)
    rev = []
    for i in range(len(v)):
        rev_i = sv[len(v)-i-1]*i
        rev.append(rev_i)

    REV=np.array(rev)
    return(rev , np.max(REV) , np.argmax(REV))



def market_clear(bids):
        
    bids = bids.tolist()
    sorted_bids = sorted(bids)
    accepted_bids = sorted_bids[len(bids)- constant_supply : ]
    p_c = min(accepted_bids)
    max_bid_value = max(accepted_bids)
    accepted_bids_indices = [[i for i, x in enumerate(bids) if x == j] for j in accepted_bids]
    final_indices = []    
    for i in accepted_bids_indices:
        x = random.choice(i)
        if x not in final_indices:
            final_indices.append(x)
        else:
            i.remove(x)
            x=random.choice(i)
            final_indices.append(x)

    return(int(p_c) , np.mean(accepted_bids) , np.sum(accepted_bids) , final_indices)
# market clearing price , average accepted bids , miner revenue , indices of the accepted bids

P_C=[]
REV=[]
p_c_old = 0
rev_old = 0
ave_bidds=[]

for x in BS:

    #start the game
    constant_supply = x


    for i in range(0 , n_iter):
        if i%1000 ==0:
            print (x , i)
        
        epsilon = 1000*0.999989**i

        
        #bidders start bidding 
        for j in range(n_bidders):
            bids[j] = take_action(j , p_c_old)
        bids = np.array(bids)
        
        # market clears
        p_c , ave_bids, rev_new , indices = market_clear(bids)
        
        #set the rewards AND update q-values
        for k in range(n_bidders):
            if k in indices:
                reward = bidders_mv[k] - bids[k]
            else:
                reward = 0

            bidder_maxQ = np.max(qval_bidders[k][: , p_c])
            qval_bidders[k][int(bids[k]) , p_c_old] = (1-alpha)*qval_bidders[k][int(bids[k]) , p_c_old] + alpha*(reward + gamma*bidder_maxQ)


        P_C.append(p_c)
        
        REV.append(rev_new)
        ave_bidds.append(ave_bids)
        
        p_c_old = p_c
        rev_old = rev_new
    
            
        
PC = moving_average(P_C , 100)

REVENUE = moving_average(REV , 100)
AVERAGE_BIDDERS = moving_average(ave_bidds , 100)
x_axes = moving_average( list(range(0 , n_iter * len(BS))) , 100)

plt.plot(x_axes , PC , 'b' , label='pc')
#plt.plot(x_axes , SUP , 'g' , label='block size')
#plt.plot(x_axes , REVENUE , 'r' , label = 'miner revenue')
#plt.plot(x_axes , AVERAGE_BIDDERS , 'y' , label = 'average accepted bids')
plt.legend()
plt.grid()
plt.show()





'''
def PLOT_PC():
    plt.plot(x_axes , PC , 'b' , label='market clearing price')
    plt.axvline(x=71, color='k', linestyle='--')
    plt.axvline(x=7, color='k', linestyle='--')
    plt.axvline(x=56, color='k', linestyle='--')
    plt.axvline(x=88, color='k', linestyle='--')
    plt.axvline(x=4, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.axvline(x=16, color='k', linestyle='--')
    plt.axvline(x=25, color='k', linestyle='--')
    plt.axvline(x=32, color='k', linestyle='--')
    plt.axvline(x=39, color='k', linestyle='--')
    plt.axvline(x=12, color='k', linestyle='--')
    plt.axvline(x=48, color='k', linestyle='--')
    plt.axvline(x=10, color='k', linestyle='--')
    plt.axvline(x=4, color='k', linestyle='--')
    plt.axvline(x=3, color='k', linestyle='--')
    plt.axvline(x=40, color='k', linestyle='--')
    plt.axvline(x=14, color='k', linestyle='--')
    plt.axvline(x=4, color='k', linestyle='--')
    plt.axvline(x=12, color='k', linestyle='--')
    plt.axvline(x=30, color='k', linestyle='--')
    plt.xlabel('number of iterations')
    plt.ylabel('market clearing price') 
    plt.legend()
    plt.grid()
    plt.show()
    



def PLOT_REV_2():
	    plt.plot(x_axes , REVENUE , 'r' , label='miner revenue')
	    plt.plot(a , b , 'b' , label='equilibrium miner revenue')
	    plt.plot(a , est_rev , 'g' , linestyle='--' , label='miner revenue after the end of each episode')
	    plt.axvline(x=y, color='k', linestyle='--')
	    plt.axvline(x=2*y, color='k', linestyle='--')
	    plt.axvline(x=3*y, color='k', linestyle='--')
	    plt.axvline(x=4*y, color='k', linestyle='--')
	    plt.axvline(x=5*y, color='k', linestyle='--')
	    plt.axvline(x=6*y, color='k', linestyle='--')
	    plt.axvline(x=7*y, color='k', linestyle='--')
	    plt.axvline(x=8*y, color='k', linestyle='--')
	    plt.axvline(x=9*y, color='k', linestyle='--')
	    plt.axvline(x=10*y, color='k', linestyle='--')
	    plt.axvline(x=11*y, color='k', linestyle='--')
	    plt.axvline(x=12*y, color='k', linestyle='--')
	    plt.axvline(x=13*y, color='k', linestyle='--')
	    plt.axvline(x=14*y, color='k', linestyle='--')
	    plt.axvline(x=15*y, color='k', linestyle='--')
	    plt.axvline(x=16*y, color='k', linestyle='--')
	    plt.axvline(x=17*y, color='k', linestyle='--')
	    plt.axvline(x=18*y, color='k', linestyle='--')
	    plt.axvline(x=19*y, color='k', linestyle='--')
	    plt.axvline(x=20*y, color='k', linestyle='--')
	    plt.xlabel('number of iterations')
	    plt.ylabel('revenue') 
	    plt.legend()
	    plt.show()




def PLOT_AVERAGE():
	    plt.plot(x_axes , AVERAGE_BIDDERS , 'y' , label='average accepted bids')
	    plt.plot(x_axes , PC , 'b' , label='market clearing price')
	    plt.axvline(x=y, color='k', linestyle='--')
	    plt.axvline(x=2*y, color='k', linestyle='--')
	    plt.axvline(x=3*y, color='k', linestyle='--')
	    plt.axvline(x=4*y, color='k', linestyle='--')
	    plt.axvline(x=5*y, color='k', linestyle='--')
	    plt.axvline(x=6*y, color='k', linestyle='--')
	    plt.axvline(x=7*y, color='k', linestyle='--')
	    plt.axvline(x=8*y, color='k', linestyle='--')
	    plt.axvline(x=9*y, color='k', linestyle='--')
	    plt.axvline(x=10*y, color='k', linestyle='--')
	    plt.axvline(x=11*y, color='k', linestyle='--')
	    plt.axvline(x=12*y, color='k', linestyle='--')
	    plt.axvline(x=13*y, color='k', linestyle='--')
	    plt.axvline(x=14*y, color='k', linestyle='--')
	    plt.axvline(x=15*y, color='k', linestyle='--')
	    plt.axvline(x=16*y, color='k', linestyle='--')
	    plt.axvline(x=17*y, color='k', linestyle='--')
	    plt.axvline(x=18*y, color='k', linestyle='--')
	    plt.axvline(x=19*y, color='k', linestyle='--')
	    plt.axvline(x=20*y, color='k', linestyle='--')
	    plt.xlabel('number of iterations')
	    plt.ylabel('bids') 
	    plt.legend()
	    plt.show()



def pc_difference():
	for i in range(1,len(BS)):
		average_pc = statistics.mean(P_C[i*800000 - 10000:i*800000])
		std_pc = statistics.stdev(P_C[i*800000 - 10000:i*800000])
		MEAN.append(average_pc)
		STDDEV.append(std_pc)


with open('passive_data.json', 'w') as jsonfile:
    json.dump(DATA_FIN, jsonfile)





'''
















