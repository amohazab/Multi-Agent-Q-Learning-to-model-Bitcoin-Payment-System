import numpy as np
import pandas as pd
import openpyxl
import random
import sys
import json
import statistics




# Define parameters:
gamma =  0.85
alpha = 0.1


bidders_mv = [71,7, 56, 88,4,1,16, 25, 32, 39, 12, 48, 10,4, 3,40,14,4,12,30]
#bidders_mv = [18 , 16 , 12 ,11 , 10, 4, 3, 2, 2, 1]
n_bidders =len(bidders_mv)

max_rev = np.sum(bidders_mv)
n_iter =3000000 #1,000,000

n_actions = np.max(bidders_mv)


#miners str
supply_lb = 0
supply_ub = 15
supply = list(range(supply_lb , supply_ub))


miner_actions = supply_ub - supply_lb + 1
#miner_states = 180
        

# Define q-value table for the users/bidders

qval_bidders = [np.zeros([int(bidders_mv[i])+1 , miner_actions , int(np.max(bidders_mv))+1]) for i in range(0,n_bidders)]

# qval_bidders: actions, previous block size , previous market clearing price

qval_miner = np.zeros([miner_actions , miner_actions , int(np.max(bidders_mv)) + 1 , max_rev + 1  , int(np.max(bidders_mv))+1])
# qval-bidders: actions , previous block size , previous market clearing price, previous revenue


# marginal values of the bidders:


bids = np.zeros(n_bidders)


# define the take_action function for a specific bidder:
def take_action(b , Supp , pc): #b: bidder index/id AND state: market clearing price
    if random.random() < epsilon:
        action = random.randint(0 , bidders_mv[b])
    else:
        action = np.argmax(qval_bidders[b][: , Supp , int(pc)])
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



def market_clear(bids , new_supply):
    if new_supply ==0:
        new_supply = 1
        
    bids = bids.tolist()
    sorted_bids = sorted(bids)
    accepted_bids = sorted_bids[len(bids)- new_supply : ]
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

    return(int(p_c) , np.mean(accepted_bids) , np.sum(accepted_bids) , final_indices  , int(max_bid_value))
# market clearing price , average accepted bids , miner revenue , indices of the accepted bids
    
    

# miners take action function
def miner_take_action(Supp , pc , rev , maxbid):
    if random.random() < epsilon:
        action = random.randint(0 , supply_ub)
    else:
        action = np.argmax(qval_miner[: , Supp , int(pc) , rev , int(maxbid)])
    return int(action)




#start the game
P_C=[]
SUPPLY=[]
REV=[]
p_c_old = 0
supply_old = 0
rev_old = 0
ave_bidds=[]
max_bid_prev = 0


for i in range(0 , n_iter):
    
    epsilon = 0.999997**i

    
    
    supply_new = miner_take_action(supply_old , p_c_old , int(rev_old) , max_bid_prev)
    
    #bidders start bidding 
    for j in range(n_bidders):
        bids[j] = take_action(j , supply_old , p_c_old)
    bids = np.array(bids)
    
    # market clears
    p_c , ave_bids, rev_new , indices  , max_bid = market_clear(bids , supply_new)
    
    #set the rewards AND update q-values
    for k in range(n_bidders):
        if k in indices:
            reward = bidders_mv[k] - bids[k]
        else:
            reward = 0

        bidder_maxQ = np.max(qval_bidders[k][: , supply_new , p_c])
        qval_bidders[k][int(bids[k]) , supply_old , p_c_old] = (1-alpha)*qval_bidders[k][int(bids[k]) , supply_old , p_c_old] + alpha*(reward + gamma*bidder_maxQ)


    miner_maxQ = np.max(qval_miner[: , supply_new , p_c , int(rev_new) , max_bid])
    miner_reward = rev_new
    qval_miner[supply_new , supply_old , p_c_old , int(rev_old) , max_bid_prev] = (1-alpha)*qval_miner[supply_new , supply_old , p_c_old , int(rev_old) , max_bid_prev] + alpha*(miner_reward + gamma*miner_maxQ)

    P_C.append(p_c)
    SUPPLY.append(supply_new)
    REV.append(rev_new)
    ave_bidds.append(ave_bids)
    
    p_c_old = p_c
    supply_old = supply_new
    rev_old = rev_new
    max_bid_prev = max_bid
    
    
    
final_pc = statistics.mean(P_C[len(P_C)-5000:])
final_supply = statistics.mean(SUPPLY[len(SUPPLY)-5000:])

res = {}
res[0] = [final_pc , final_supply]
print(res)



















