You can find the full version of the paper in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3974688

eps_greed_github.py: It is a python file that include the main logic. We tried to model the Bitocin Payment System with the multi-uint dynamic auction framework. Since we wanted to solve the problem without any restricting assumption, we used the multi-agent Q-learning algorithm. In this case, all of the agents, bidders and miner(assumption of monopoly), are AI agent. The miner sets the number of items to sell and the bidders bid for the available slots. In this q-learning algorithm, states are the information from the previous state, actions are bidding for the users and choosing the number of transaction to send for the miner. Finally, rewards are the difference between the marginal valuation and bid values for the users and the sum of tramsaction fees for the miner.

PASSIVE.py: Since we did not get convergence in the previous case, we tried to put restriction on the model so that the miner is not AI, but he tries to play one specific strategy for a long time to obtain the equilibrium plays of the players. Considering the equilibrium response of all the users, The miner will maximise his profit by choosing the optimal action. The final results are promissing!

Sync_constant.py: PASSIVE format with the difference that now we use Synchronous version of agents learning. For more info about Synchronous learning please check: Artificial Intelligence and Pricing: The Impact of Algorithm Design (2021) by Asker, J., Fershtman, C., & Pakes, A.

Data: To derive the actual evidence of manipulaiton from the historical data, we used two sources. First, we gathered the block-specific data from https://blockchair.com/ which is a valuable resource for collecting information about blocks and transactions for different blockchains such as Bitcoin, Ethereum, litecoin and etc. Second, we used the jochen hoenicke website, https://mempool.jhoenicke.de/#BTC,2h,weight to gather the information about mempool size. It is an interesting website which presents the diverse set of information about mempool such as the size, fee and the number of the transactions waiting to be confirmed. 
