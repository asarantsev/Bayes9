# This is the file where we have 5 asset classes:
# 3 geometric stock returns: S&P, developed, emerging
# 2 arithmetic bond returns: Corporate investment-grade and Treasury
# There are 4 factors:
# stock valuation measure and volatility
# BAA and Treasury bond rates
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
DF = pd.read_excel('9model.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
price = DF['Price'].values
div = DF['Dividends'].values
rates = DF['BAA'].values
long = DF['Treasury'].values
bonds = DF['Bonds'].values[45:]
intl = DF['International'].values[43:]
em = DF['Emerging'].values[61:]
zeros = DF['Zeros'].values[35:]
N = 98 # overall number of data points

# total returns
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N)])
wealth = np.exp(np.append(np.array([0]), np.cumsum(total)))
premeasure = np.log(wealth/div) # measure before detrending

# regression equation for computation of the valuation measure
measureReg = OLS(np.diff(premeasure), pd.DataFrame({'const' : 1, 'trend' : np.array(range(N)), 'slope' : premeasure[:-1]})).fit()
print('regression to create valuation measure')
# print(measureReg.summary())
measure = premeasure + measureReg.params['trend']/measureReg.params['slope'] * range(N + 1)

# Fitting autoregression for the valuation measure with stochastic volatility
RegMeasure = OLS(np.diff(measure)/vol, pd.DataFrame({'const' : 1/vol, 'lag' : measure[:-1]/vol, 'vol' : 1})).fit()

# Regression for the USA normalized geometric stock returns
# versus duration and valuation measure, but without risk spread
mainDF = pd.DataFrame({'const' : 1/vol, 'duration' : -np.diff(rates)/vol, 'measure' : -measure[:-1]/vol, 'vol' : 1})
RegUSA = OLS(total/vol, mainDF).fit()

# Regression for geometric normalized returns of international developed stocks
# versus duration and valuation measure, but without risk spread
nIntlRet = np.log(np.ones(56) + intl)/vol[42:]
RegIntl = OLS(nIntlRet, mainDF.iloc[42:]).fit()

# Cut regression for normalized geometric returns of emerging markets
# with duration but without the valuation measure or spreads
nEMRet = np.log(np.ones(38) + em)/vol[60:]
RegEM = OLS(nEMRet, mainDF.iloc[60:]).fit()

# Autoregression of corporate bond rates with stochastic volatility
RegBondRates = OLS(np.diff(np.log(rates))/vol, pd.DataFrame({'const' : 1/vol, 'lag' : np.log(rates)[:-1]/vol})).fit()

# apply Yeo-Johnson transform to log volatility and fit autoregression for transformed volatility
lvol = np.log(vol)
nvol = scipy.stats.yeojohnson(lvol, -0.5)
RegVol = OLS(np.diff(nvol), pd.DataFrame({'const' : 1, 'lag' : nvol[:-1]})).fit()

# Arithmetic corporate bond returns vs rates regression with stochastic volatility
RegBondReturns = OLS(np.log(bonds[1:]/bonds[:-1] - 0.01 * rates[45:-1])/vol[45:], pd.DataFrame({'const' : 1, 'duration' : -np.diff(rates)/vol}).iloc[45:]).fit()

# Autoregression of risk log spread of logs
spreads = np.log(rates) - np.log(long)
lspreads = np.log(spreads)
RegRiskSpreads = OLS(np.diff(lspreads), pd.DataFrame({'const' : 1, 'lag' : lspreads[:-1]})).fit()

# Regression of 7-year zero-coupon Treasury bond returns upon the changes in 10-year Treasury rates
benchmark = (7 * long[34:-1] - 6 * long[35:]) * 0.01 
RegZerosReturns = OLS(zeros, pd.DataFrame({'const' : 1, 'benchmark' : benchmark})).fit()

# print the covariace and correlation matrix for residuals, 
# there are 9 series since 4 + 5 = 9 equations
allReg = [RegUSA, RegIntl, RegEM, RegBondReturns, RegVol, RegBondRates, RegMeasure, RegRiskSpreads, RegZerosReturns]
allResid = [Reg.resid for Reg in allReg] 
lengths = [len(res) for res in allResid]
allNames = ['usa', 'intl', 'em', 'bond-ret', 'trans-vol', 'bond-rates', 'measure', 'spreads', 'zeros']
allResiduals = pd.DataFrame(columns = allNames)

for k in range(9):
    print(allNames[k])
    Reg = allReg[k]
    # print(Reg.summary())
    # print(Reg.params)
    beta_hat = Reg.params.to_numpy() # OLS coefficient estimate
    print('point estimates = ', beta_hat)
    covBayes = Reg.normalized_cov_params.to_numpy() * Reg.mse_resid # covariance matrix
    print('covariance Bayesian matrix = ')
    print(covBayes)
    # print('number of residuals = ', len(allResid[k]))
    # verification(allResid[k])
    allResiduals[allNames[k]] = np.pad(allResid[k], (N - lengths[k], 0), constant_values = np.nan)
    
covMatrix = allResiduals.cov()
# we print it row by row since the screen does not fit the entire 9x9 matrices
print('Covariance Matrix')
for name in allNames:
    print(covMatrix[name].values*10000)
