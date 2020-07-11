# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:01:30 2020

@author: Amogh


                             ##########   PROBLEM   #########
                                        
The Investment Fund ABC currently has a ₹1,000,000  position in the Nifty 50 Index.
The Risk Manager of the Fund wants to estimate the tail risk (extreme negative outcomes) of this position based on historical data (and forecasts).
Simulate the minimum loss over a period of One Quarter that will occur with 1% probability:
    1% Value-at-Risk (VaR) of ₹1,000,000 over a period of one quarter (63 business days).

Here, we attempt to calculate VaR using the Parametric Method (Monte Carlo Simulation).

"""

import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

stocks = ["^NSEI"]
start = dt.datetime.today()-dt.timedelta(5500)
end = dt.datetime.today()
ohlcv_data = {}

# Downloading NIFTY price data and calculating daily returns from September 2007 until now:
for ticker in stocks:
    ohlcv_data[ticker] = yf.download(ticker,start,end).dropna(how="all")

nifty_ret = ohlcv_data["^NSEI"]["Adj Close"].pct_change(1).dropna()
nifty_ret

# Checking Mean and Standard Deviation of the returns:
mean = nifty_ret.mean()
std = nifty_ret.std()
print(mean,std)

# Defining VaR Parametric:
def var_param(returns, days, prob, I0, sims):
    mean = np.mean(returns)
    std = np.std(returns)
    days = int(days)
    
    ret = np.random.normal(loc = mean, scale = std, size = days * sims).reshape(sims, days)
    
    final = (ret + 1).prod(axis = 1) * I0
    var = np.percentile(final, prob) - I0
    
    return var


# As per our problem:
I0 = 1000000     # ₹1M position
sims = 1000000   # no.of simulations we are running
prob = 1         # 1% VaR (probability)
days = 63        # 1 quarter = 252 trading days/4 = 63 days 


# Calculating VaR:
var_param(nifty_ret, days, prob, I0, sims)


#########   Calculating and Visualizing the relationship between TIME PERIOD and VaR   ###########

# Calculating the VaR for many time periods:
var_p1 = []
for i in range(1, 252+1):
    var_p1.append(-var_param(nifty_ret, days = i, prob = 1, I0 = 1000000, sims = 10000))

# Visualization:
plt.figure(figsize = (4, 3))
plt.plot(range(1, 252+1), var_p1)
plt.title("1% VaR for ₹1,000,000", fontsize = 20)
plt.xlabel("Period (in days)", fontsize = 15)
plt.ylabel("VaR", fontsize = 15)
plt.show()

# COMMENT: No surprise that the VaR increases as the Time Period increases. Is is Not a Linear Relationship.


#########   Calculating and Visualizing the relationship between VaR and PROBABILITY  ###########

# Calculating the VaR given probabilities between 0.1% and 5%:
var_p2 = []
for i in np.linspace(0.1, 5, 100):
    var_p2.append(-var_param(nifty_ret, days = 63, prob = i, I0= 1000000, sims = 10000))

# Visualization:
plt.figure(figsize = (4, 3))
plt.plot(np.linspace(0.1, 5, 100), var_p2)
plt.title("Quarterly VaR for ₹1,000,000", fontsize = 20)
plt.xlabel("Probability (in%)", fontsize = 15)
plt.ylabel("VaR", fontsize = 15)
plt.show()

# COMMENT: Lower the probability, more extreme are the outcomes(Higher VaR). 


