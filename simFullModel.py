# The main simulation function with the complete 9-series model
# 5 asset classes returns, all arithmetic nominal annual total:

# USA stocks measured by Standard & Poor
# developed ex-USA stocks, measured by MSCI EAFE 88% + MSCI Canada 12%
# emerging markets stocks, measured by MSCI EM
# USA long-term Treasuries, measured by 10-year zero-coupon Treasury bonds
# USA investment-grade corporate bonds, measured by MSCI index

# 4 market factors:

# stock volatility V
# valuation measure H
# BAA bond rates R
# 10-year Treasury rates L 

# We use the valuation measure based on 1-year dividends
# initialF = initial factor (F = V, H, L, R)
# T is time horizon in years
# returns five 2d arrays, each has rows which are time series simulations

# geometric nominal returns
# domestic stocks Q
# developed international stocks I
# and emerging stocks E

# arithmetic bond returns
# long-term Treasury A
# and investment-grade corporate bonds B

# Equations for geometric G stock returns, arithmetic stock returns are computed as e^G - 1

# Domestic: Q(t) = 0.26369 - 0.012892 *V(t) - 0.05532 * (R(t)-R(t-1)) - 0.13033 * H(t-1) + V(t) * Z_Q(t)

# Developed: I(t) = 0.26837 - 0.018021 * V(t) - 0.039046 * (R(t) - R(t-1)) + V(t) * Z_I(t)

# Emerging: E(t) = 0.024347 - 0.014995 * V(t) - 0.091255 * (R(t) - R(t-1)) + V(t) * Z_E(t)

# Equations for arithmetic bond returns

# Investment-grade corporate bonds
# B(t) = 0.01 * R(t-1) + exp(-0.015208 - 0.036084 * (R(t) - R(t-1)) + Z_B(t)) - 1

# Treasury long-term (10-year) bonds
# A(t) = (L(t-1)/L(t))*(1 - (1 + L(t))**(-10)) + (1 + L(t))**(-10) - 1 + L(t-1)

# Next, the four factors are:
# with dC(t) = C(t) - C(t-1)
# and F the Yeo-Johnson transform with index -0.5

# Stock market volatility
# dF(ln V(t)) = - 0.39913 * F(ln V(t-1)) + 0.35953 + W_V(t)

# BAA rate
# d(ln R(t)) = -0.0004159 + V(t) * W_R(t)

# Spread of logs S(t) = 0.1*ln(ln R(t) - ln L(t))
# dS(t) = - 0.023448 * S(t-1) - 0.024614 + W_S(t)

# Valuation measure
# dH(t) = 0.16986 - 0.17381 * H(t-1) - 0.01291 + V(t) * W_H(t)

import numpy
from scipy.stats import yeojohnson

NSIMS = 10000 # number of simulations

Sigma = 0.0001 * numpy.array([[2.02636925,  0.84770263, -0.07913848, -0.34926746, -0.61493991, -0.11569614,  2.06356837, -0.87098483], [ 0.84770263,  2.97529756,  2.18081193, -1.38653027, -0.85514725, 0.00605395, 0.863375, -2.14066108], [-0.07913848,  2.18081193,  6.20397705, -2.54267855, -1.94505869, -0.02943804, -0.04497621, -3.56895948], [-0.34926746, -1.38653027, -2.54267855, 13.71620885,  6.01738781, -0.85044526, -0.05811848, 10.00467142], [-0.61493991, -0.85514725, -1.94505869,  6.01738781, 39.19824138, 2.7337162, -1.174323, 10.14939623], [-0.11569614,  0.00605395, -0.02943804, -0.85044526,  2.7337162, 1.99958699, -0.84030497, -0.5201592], [ 2.06356837,  0.863375,   -0.04497621, -0.05811848, -1.174323, -0.84030497, 3.15243454, -0.57366691], [-0.87098483, -2.14066108, -3.56895948, 10.00467142, 10.14939623, -0.5201592, -0.57366691, 14.87541169]])

# Inverse Yeo-johnson transform with lambda = -0.5
def YJinv(y):
    y = numpy.asarray(y, dtype = float)
    x = numpy.empty_like(y)
    pos = y >= 0
    neg = ~pos
    # y >= 0: y = ((x + 1)^lambda - 1) / lambda
    x[pos] = numpy.power(1 + (-0.5) * y[pos], -2) - 1
    # y < 0: y = -[((1 - x)^(2-lambda) - 1) / (2-lambda)]
    x[neg] = 1 - numpy.power(1 - (2.5) * y[neg], 0.4)
    return x

# returns of coupon Treasury bonds
def zeroRet(last, new):
    return (last/new)*(1 - (1 + new)**(-10)) + (1 + new)**(-10) - 1 + last

# This is the main simulation function
# T = time horizon
def sim(initialV, initialH, initialR, initialL, T):
    
     # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(8), Sigma, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns Z_Q
    noiseIntl = noise[:, :, 1] # international developed stock returns Z_I
    noiseEm = noise[:, :, 2] # emerging stock returns Z_E
    noiseBonds = noise[:, :, 3] # corporate bond returns Z_B
    
    noiseVol = noise[:, :, 4] # volatility W_V
    noiseRates = noise[:, :, 5] # corporate bond rates W_R
    noiseMeasure = noise[:, :, 6] # the new valuation measure W_H
    noiseSpreads = noise[:, :, 7] # the log spreads of logs W_S
    
    # now initialize the 2d arrays corresponding to simulated series
    simRetUSA = numpy.zeros((T, NSIMS))
    simRetIntl = numpy.zeros((T, NSIMS))
    simRetEm = numpy.zeros((T, NSIMS))
    simRetBonds = numpy.zeros((T, NSIMS))
    simRetLong = numpy.zeros((T, NSIMS))
    
    simTVol = numpy.zeros((T+1, NSIMS))
    simLRates = numpy.zeros((T+1, NSIMS))
    simLSpreads = numpy.zeros((T+1, NSIMS))
    simMeasure = numpy.zeros((T+1, NSIMS))
        
    # initialize some simulated series given initial conditions
    simTVol[0] = yeojohnson(numpy.log(initialV), -0.5) * numpy.ones(NSIMS)
    simLRates[0] = numpy.log(initialR) * numpy.ones(NSIMS)
    simMeasure[0] = initialH * numpy.ones(NSIMS)
    simLSpreads[0] = 0.1 * numpy.log(numpy.log(initialR) - numpy.log(initialL)) * numpy.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simTVol[t + 1] = 0.35953 * numpy.ones(NSIMS) + (1 - 0.39913) * simTVol[t] + noiseVol[t]
        
    # take exponents to get volatility
    simVol = numpy.exp(YJinv(simTVol))
    
    # simulate log rates as heteroscedastic random walk
    for t in range(T):
        simLRates[t + 1] = simLRates[t] - 0.0004159 * numpy.ones(NSIMS) + noiseRates[t] * simVol[t + 1] 
        
    # take exponents to get rates
    simRates = numpy.exp(simLRates)
    
    # simulate the valuation measure as autoregression with stochastic volatility
    for t in range(T):
        simMeasure[t + 1] = 0.16986 + (1 - 0.17381) * simMeasure[t] - 0.01291 * simVol[t + 1] + simVol[t + 1] * noiseMeasure[t]

    # simulate the log spread as autoregression
    for t in range(T):
        simLSpreads[t + 1] = (1 - 0.023448) * simLSpreads[t] - 0.024614 * numpy.ones(NSIMS) + noiseSpreads[t]
        
    # simulate the long-term Treasury rate
    # We rewrite the equation S(t) = ln (R(t)) - ln (L(t))
    # as L(t) = exp(ln R(t) - S(t))
    simLong = numpy.exp(simLRates - numpy.exp(10 * simLSpreads))

    
    # simulate arithmetic stock returns 
    for t in range(T):
        # three series of stock returns
        simRetUSA[t] = numpy.exp(0.26369 * numpy.ones(NSIMS) - 0.012892 * simVol[t+1] - 0.055320 * (simRates[t+1] - simRates[t]) - 0.1440 * simMeasure[t] + simVol[t + 1] * noiseUSA[t]) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(0.26837 * numpy.ones(NSIMS) - 0.018021 * simVol[t+1] - 0.039046 * (simRates[t+1] - simRates[t]) - 0.0641 * simMeasure[t] + simVol[t+1] * noiseIntl[t]) - numpy.ones(NSIMS)
        simRetEm[t] = numpy.exp(0.024347 * numpy.ones(NSIMS) - 0.014995 * simVol[t+1] - 0.091255 * (simRates[t+1] - simRates[t]) + 0.1155 * simMeasure[t] + simVol[t + 1] * noiseEm[t]) - numpy.ones(NSIMS)
        
        # two series of bond returns
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(-0.015208 * numpy.ones(NSIMS) - 0.036084 * (simRates[t+1] - simRates[t]) + noiseBonds[t]) - numpy.ones(NSIMS)
        simRetLong[t] = zeroRet(0.01 * simLong[t], 0.01 * simLong[t+1])    
    return [simRetUSA, simRetIntl, simRetEm, simRetLong, simRetBonds]
