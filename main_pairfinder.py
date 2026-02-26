import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import os
import time 
#import matplotlib.pyplot as plt

CACHE_FILE = "sp500_data_cache.csv"
CACHE_EXPIRY_SECONDS = 900
#step 1: import data of ETF (closing price)
if os.path.exists(CACHE_FILE) and (time.time() - os.path.getmtime(CACHE_FILE)) < CACHE_EXPIRY_SECONDS:
    print("Loading recent market data from local cache...")
    data = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    print("Downloading fresh market data from Yahoo Finance...")
    # step 1: import data of ETF (closing price)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    snpdata = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})
    sp500 = snpdata[0]

    assets = sp500['Symbol'].tolist()
    assets = [asset.replace('.','-') for asset in assets]
    
    data = yf.download(assets, start="2020-01-01")['Close']
    data.to_csv(CACHE_FILE)


if __name__ == '__main__':
    #step 2: make all possible combinations 
    tickers= data.columns
    pairs=pd.DataFrame(combinations(tickers, 2), columns=['Stock 1', 'Stock 2'])
    #print(pairs)

    #step 3:tabulate this information as |stock1|stock2|correlation| 
    corr_matrix = data.corr().values
    ticker_to_idx = {ticker: i for i, ticker in enumerate(data.columns)}

    idx1 = pairs['Stock 1'].map(ticker_to_idx).values
    idx2 = pairs['Stock 2'].map(ticker_to_idx).values
    pairs['Correlation'] = corr_matrix[idx1, idx2]

    #step 4:filter for highly correlated stocks
    filteredpairs= pairs[pairs['Correlation']>=0.95] #filtered for pairs with correlcoeff higher or equal to 0.95
    #print(len(filteredpairs['Correlation']))

    #step 5:find regression and then apply ADF test to these residuals &
    #step 6:filter out based on p value to reject the null hypothesis (<0.01) &
    #step 7:critical values &
    #step 8:adf test statisitic should be lower than critical value (1%)

    cointpairs=[]
    for index, row in filteredpairs.iterrows():
        s1= data[row['Stock 1']]
        s2= data[row['Stock 2']]
        # we need regression in the form S2= alpha+(beta*S1)+Epsilon where beta is hedge ratio, alpha is the intercept and epislon is the spread 
        X=sm.add_constant(s1) #adds intercept (alpha)
        model = sm.OLS(s2, X, missing='drop').fit()
        beta=model.params.iloc[1] #determines beta from model it makes in above line; params[0] would be the alpha
        spread = model.resid.dropna() #deterimines epsilon
        adfresult = adfuller(spread) #does the adf test
        adfstat = adfresult[0]
        pvalue=adfresult[1]
        critvalues = adfresult[4]
        if pvalue < 0.01:
            cointpairs.append({
                'Stock 1': row['Stock 1'],
                'Stock 2': row['Stock 2'],
                'Beta': beta,
                'ADF stat': adfstat, 
                'Crit value': critvalues['1%']
            })

    #step 9:tabluate results of pairs accoring to the most negative test statistic

    finalpairs=pd.DataFrame(cointpairs)
    finalpairs=finalpairs.sort_values(by='ADF stat')
    print("\n" + "="*50)
    print(finalpairs.head(10))