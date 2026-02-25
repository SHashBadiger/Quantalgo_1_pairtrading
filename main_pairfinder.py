import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

#step 1: import data of ETF (closing price)
url= 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
snpdata = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})
sp500=snpdata[0]

assets= sp500['Symbol'].tolist()

assets= [asset.replace('.','-')for asset in assets]
data = yf.download(assets, start="2020-01-01")['Close']# this close is automatically adj close.
#print(data)



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
print(len(filteredpairs['Correlation']))
#step 5:apply ADF test to these stocks
#step 6:filter out based on p value to reject the null hypothesis (<0.01)
#step 7:critical values 
#step 8:adf test statisitic should be lower than critical value (1%)
#step 9:tabluate results of pairs accoring to the most negative test statistic

