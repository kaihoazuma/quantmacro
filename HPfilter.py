import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

UK_GDP_SYMBOL = 'NGDPRSAXDCGBQ'  
JP_GDP_SYMBOL = 'JPNRGDPEXP'  
start_date = '1994-01-01'
end_date = '2022-01-01'


uk_gdp = web.DataReader(UK_GDP_SYMBOL, 'fred', start_date, end_date)
log_uk_gdp = np.log(uk_gdp)
uk_cycle, _ = sm.tsa.filters.hpfilter(log_uk_gdp, lamb=1600)
uk_std = uk_cycle.std()

jp_gdp = web.DataReader(JP_GDP_SYMBOL, 'fred', start_date, end_date)
log_jp_gdp = np.log(jp_gdp)
jp_cycle, _ = sm.tsa.filters.hpfilter(log_jp_gdp, lamb=1600)
jp_std = jp_cycle.std()

correlation = np.corrcoef(uk_cycle, jp_cycle)[0, 1]


plt.figure(figsize=(10, 6))
plt.plot(log_uk_gdp.index, uk_cycle, label="UK GDP Cycle (trend removed)")
plt.plot(log_jp_gdp.index, jp_cycle, label="Japan GDP Cycle (trend removed)")
plt.legend()
plt.title("UK and Japan GDP")
plt.xlabel("Year")
plt.ylabel("Log GDP")
plt.show()


print("UK GDP standard deviation:", uk_std)
print("Japan GDP standard deviation:", jp_std)
print("Correlation coefficient between UK and Japan GDP:", correlation)

