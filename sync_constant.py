import numpy as np
import pandas as pd
import openpyxl
import random
import sys
import json

GAMMA = [0.05 , 0.1 , 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
PARAMS={}
count = 0
for i in range(len(GAMMA)):
    PARAMS[str(count)] = GAMMA[i]
    count = count + 1


# Define parameters:
gamma =  PARAMS[sys.argv[1]]
alpha = 0.1



bidders_mv = [71,7, 56, 88,4,1,16, 25, 32, 39, 12, 48, 10,4, 3,40,14,4,12,30]
#bidders_mv = [18 , 16 , 12 ,11 , 10, 6, 3, 2, 2, 1]
n_bidders =len(bidders_mv)

max_rev = np.sum(bidders_mv)
n_iter =3000000 #1,000,000

n_actions = np.max(bidders_mv)

constant_supply = 7

        

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


#start the game
P_C=[]
SUPPLY=[]
REV=[]
p_c_old = 0
supply_old = 0
rev_old = 0
ave_bidds=[]



for i in range(0 , n_iter):
    if i%1000 ==0:
        print (i)
    
    epsilon = 1000*0.999997**i

    
    #bidders start bidding 
    for j in range(n_bidders):
        bids[j] = take_action(j , p_c_old)
    bids = np.array(bids)
    
    # market clears
    p_c , ave_bids, rev_new , indices = market_clear(bids)
    
    #set the rewards AND update q-values
    for k in range(n_bidders):
        bidder_maxQ = np.max(qval_bidders[k][: , p_c])
        for p in range(bidders_mv[k]):
            if p > p_c:
                reward = bidders_mv[k] - p
            else:
                reward = 0
                
            qval_bidders[k][p , p_c_old] = (1-alpha)*qval_bidders[k][p , p_c_old] + alpha*(reward + gamma*bidder_maxQ)


    P_C.append(p_c)
    
    REV.append(rev_new)
    ave_bidds.append(ave_bids)
    
    p_c_old = p_c
    rev_old = rev_new



pc_array = P_C[len(P_C) - 20000 : len(P_C)]
# results: gamma , alpha , real_supply , real_pc , np.mean(pc_array) , np.std(pc_array) , np.mean(supply_array) , np.std(supply_array)
res = [gamma , np.mean(pc_array) , np.std(pc_array)]
print(res)

