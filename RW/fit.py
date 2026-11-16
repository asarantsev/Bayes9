import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.api import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [10])['lb_pvalue'].values)

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()

DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[10:]
price = DF['Price'].values[9:]
div = DF['Dividends'].values[9:]
rates = DF['BAA'].values[1:]
long = DF['Long'].values[7:]
bondW = DF['Bonds'].values[54:]
dev = DF['Developed'].values[52:]
emer = DF['Emerging'].values[70:]
earn = DF['Earnings'].values
cpi = DF['CPI'].values

N = 100 # maximum number of data points
W = 10 # window
total = np.array([np.log(price[k+1] + div[k+1]) - np.log(price[k]) for k in range(N-2)])
nUSAret = total/vol
nDevRet = np.log(np.ones(56) + dev)/vol[42:]
nEmerRet = np.log(np.ones(38) + emer)/vol[60:]

tvol = scipy.stats.yeojohnson(np.log(vol), -0.5)
VOLdf = pd.DataFrame({'const' : 1, 'lag' : tvol[:-1]})
VOLreg = OLS(tvol[1:], VOLdf).fit()

rearn = earn * cpi[-1]/cpi
rprice = price * cpi[-1]/cpi[W-1:]
cumearn = np.array([sum(rearn[k:k+W])/W for k in range(99)])
shield = cumearn/rprice
dshield = np.diff(np.log(shield))
ndshield = dshield/vol
SHdf = pd.DataFrame({'const': np.ones(98)})
SHreg = OLS(ndshield, SHdf).fit()

USAdf = pd.DataFrame({'vol' : 1, 'cape' : shield[:-1]/vol, 'duration' : -np.diff(rates[8:])/vol})
USAreg = OLS(nUSAret, USAdf).fit()

DEVdf = USAdf.iloc[-56:]
DEVreg = OLS(nDevRet, DEVdf).fit()

EMdf = USAdf.iloc[-38:]
EMreg = OLS(nEmerRet, EMdf).fit()

drates = np.diff(np.log(rates[8:]))
ndrates = drates/vol
RATEdf = SHdf
RATEreg = OLS(ndrates, RATEdf).fit()

bonds = np.diff(np.log(bondW))
BONDdf = pd.DataFrame({'const' : 1, 'duration' : -np.diff(rates)[53:]})
BONDreg = OLS(bonds - 0.01 * rates[53:-1], BONDdf).fit()

lspread = np.log(np.log(rates[6:]) - np.log(long))
SPREADdf = pd.DataFrame({'const' : 1, 'lag' : lspread[:-1]})
SPREADreg = OLS(np.diff(lspread), SPREADdf).fit()

# Print summaries, point estimates, stderrs,
# and Bayesian covariance matrix
allReg = [USAreg, DEVreg, EMreg, BONDreg, VOLreg, SHreg, RATEreg, SPREADreg]
allDF = [USAdf, DEVdf, EMdf, BONDdf, VOLdf, SHdf, RATEdf, SPREADdf]
allResid = [Reg.resid for Reg in allReg] 
lengths = [len(res) for res in allResid]
allNames = ['usa', 'developed', 'emerging', 'bonds', 'volatility', 'shiller', 'baa', 'spreads']
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
print('Covariance Matrix')
for name in allNames:
    print(covMatrix[name].values)
    
# we print it row by row since the screen does not fit the entire matrix
print('Correlation Matrix')
for name in allNames:
    print(corrMatrix[name].values)
