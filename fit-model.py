# This is the file where we have 5 asset classes:
# 3 geometric stock returns: S&P, developed, emerging
# 2 arithmetic bond returns: Corporate investment-grade and Treasury
# There are 4 factors:
# stock valuation measure and volatility
# BAA corporate and 10-year Treasury bond rates
# All data is annual and nominal, not inflation-adjusted
# We select chosen regressions and print them, also residual analysis

import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.api import stats

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)

# reading the data file
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[3:]
price = DF['Price'].values[2:]
div = DF['Dividends'].values[2:]
rates = DF['RateBAA'].values
long = DF['RateLong'].values
bondReturns = DF['BAAreturns'].values[1:]
dev = DF['Developed'].values[45:]
emer = DF['Emerging'].values[63:]

N = 100 # overall number of data points

# total returns
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N-2)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))
premeasure = np.log(wealth/div) # measure before detrending

# regression equation for computation of the valuation measure
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N-2)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
# print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N-1)

# Fitting autoregression for the valuation measure with stochastic volatility
RegMeasure = OLS(np.diff(measure)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()

# Regression for the USA normalized geometric stock returns
# versus duration and valuation measure, but without risk spread
RegUSA = OLS(total/vol, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)[2:]/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})).fit()

# Regression for geometric normalized returns of international developed stocks
# versus duration and valuation measure, but without risk spread

nDevRet = np.log(np.ones(56) + dev)/vol[42:]
RegDev = OLS(nDevRet, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)[2:]/vol, 'vol' : 1}).iloc[42:]).fit()

# Regression for normalized geometric returns of emerging markets
# with duration but without the valuation measure or spreads
nEmerRet = np.log(np.ones(38) + emer)/vol[60:]
RegEmer = OLS(nEmerRet, pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)[2:]/vol, 'vol' : 1}).iloc[60:]).fit()

# Autoregression of corporate bond rates with stochastic volatility
RegBAArate = OLS(np.diff(np.log(rates))[2:]/vol, pd.DataFrame({'const' : np.ones(N-2)})).fit()

# apply Yeo-Johnson transform to log volatility and fit autoregression for transformed volatility
lvol = np.log(vol)
nvol = scipy.stats.yeojohnson(lvol, -0.5)
RegVol = OLS(np.diff(nvol), pd.DataFrame({'const' : 1, 'lag' : nvol[:-1]})).fit()

# Arithmetic corporate bond returns vs rates regression with stochastic volatility
adjBAAret = np.log(bondReturns + 1 - 0.01 * rates[:-1])
RegBAAret = OLS(adjBAAret, pd.DataFrame({'const' : 1, 'duration' : -np.diff(rates)})).fit()

# Autoregression of risk log spread of logs
spreads = np.log(rates) - np.log(long)
lspreads = np.log(spreads) * 0.1
RegSpreads = OLS(np.diff(lspreads), pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()

# Analyze residuals of each of 9 regressions
# Print summaries, point estimates, stderrs,
# and Bayesian covariance matrix
allReg = [RegUSA, RegDev, RegEmer, RegBAAret, RegVol, RegBAArate, RegMeasure, RegSpreads]
allResid = [Reg.resid for Reg in allReg] 
lengths = [len(res) for res in allResid]
allNames = ['domestic', 'developed', 'emerging', 'BAA-returns', 'trans-vol', 'BAA-rates', 'measure', 'risk-spreads']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(8):
    print(allNames[k])
    Reg = allReg[k]
    print(Reg.summary())
    beta_hat = Reg.params.to_numpy() # OLS coefficient estimate
    print('point estimates = ', beta_hat)
    # covariance matrix for objective Bayes posterior
    covBayes = Reg.normalized_cov_params.to_numpy()
    print('covariance Bayesian matrix = ')
    print(covBayes)
    print('number of residuals = ', len(allResid[k]))
    verification(allResid[k]) # checking innovations are IID Gaussian
    print('stderr = ', np.std(allResid[k]))
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
# print the covariace and correlation matrix for residuals
# there are 9 series since 4 + 5 = 9 equations
covMatrix = allResiduals.cov()
corrMatrix = allResiduals.corr()

# we print it row by row since the screen does not fit the entire matrix
# we multiply by 10000 since the original residuals are of order 0.01
print('Covariance Matrix')
for name in allNames:
    print(covMatrix[name].values*10000)
    
# we print it row by row since the screen does not fit the entire matrix
print('Correlation Matrix')
for name in allNames:
    print(corrMatrix[name].values)