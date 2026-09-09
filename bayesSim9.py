# The main simulation function with the complete 9-series model
# 5 outputs: US stocks, ex-US developed stocks, emerging stocks
# and 10-year Treasury and investment-grade corporate bonds
# 4 factors: stock volatility V, and the valuation measure H
# BAA bond rates R, 10-year Treasury rates L 
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

import numpy
import scipy

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

NSIMS = 10000

# coefficient estimates and Bayesian covariance matrix for US stock returns
# order: 1/vol, -np.diff(rates)/vol, -measure[:-1]/vol, 1
pointUS  = numpy.array([0.26369255, 0.05531972, 0.13032606, -0.01289155])
covUS = (numpy.array([[ 1.20309794e-03, -8.83528968e-05, 7.70409213e-04, -1.17693913e-04],
 [-8.83528968e-05,  2.01281081e-04, -9.91713817e-05,  6.70198050e-06],
 [ 7.70409213e-04, -9.91713817e-05,  1.90810867e-03, -3.21309983e-05],
 [-1.17693913e-04,  6.70198050e-06, -3.21309983e-05,  1.49705561e-05]]))

# coefficient estimates and Bayesian covariance matrix for developed stock returns
# order: 1/vol, -np.diff(rates)/vol, -measure[:-1]/vol, 1
pointDev = numpy.array([ 0.27787639,  0.03958753,  0.0798607,  -0.01722616])
covDev = (numpy.array([[3.67447998e-03, -3.52301676e-04, 6.92192902e-04, -4.02309553e-04],
 [-3.52301676e-04, 3.87530356e-04, 3.94405888e-05, 3.54513032e-05],
 [ 6.92192902e-04, 3.94405888e-05, 5.81253354e-03, 5.78669736e-05],
 [-4.02309553e-04, 3.54513032e-05, 5.78669736e-05, 5.27845045e-05]]))

# coefficient estimates and Bayesian covariance matrix for emerging stock returns
# order: 1/vol, -np.diff(rates)/vol, -measure[:-1]/vol, 1
pointEM = numpy.array([ 0.23986833, 0.09262668, -0.03340665, -0.01555279])
covEM = (numpy.array([[ 0.01124397, -0.00177033,  0.00206245, -0.00116777],
 [-0.00177033,  0.00204362, -0.00078455,  0.00012877],
 [ 0.00206245, -0.00078455,  0.01911321,  0.00031933],
 [-0.00116777,  0.00012877,  0.00031933,  0.00015522]]))

pointBAA = numpy.array([-0.00165298,  0.06097906])
covBAA =  numpy.array([[ 2.20325613e-07, -1.87489568e-07], [-1.87489568e-07,  1.34754388e-05]])

# coefficient estimates and Bayesian covariance matrix for transformed log volatility
# order: 1, nvol[:-1] (lag)
pointNVol = numpy.array([0.3515266, 1-0.39913095])
covNVol = numpy.array([[ 0.00527463, -0.00594903], [-0.00594903, 0.00676201]])

# TBD
pointRates = numpy.array([ 0.0707774, -0.04409479])
covRates = numpy.array([[0.0026996, -0.00143464], [-0.00143464, 0.00079885]])

# coefficient estimates and Bayesian covariance matrix for the new valuation measure
# order: 1/vol, measure[:-1]/vol, 1
pointMeasure = numpy.array([ 0.16985524, -0.17381195, -0.01291036])
covMeasure = (numpy.array([[ 1.79226532e-03, -1.11890460e-03, -1.76641291e-04],
 [-1.11890460e-03,  2.86199447e-03,  4.43772314e-05],
 [-1.76641291e-04,  4.43772314e-05,  2.27011199e-05]]))

# coefficient estimates and Bayesian covariance matrix for transformed log volatility
# order: 1, lspreads[:-1] (lag)
pointLSpreads = numpy.array([-0.19795661, -0.18730095])
covLSpreads = numpy.array([[0.00495171, 0.00361859], [0.00361859, 0.00349211]])

# coefficient estimates and Bayesian covariance matrix for zeros returns
# versus the benchmark = (7 * long[34:-1] - 6 * long[35:]) * 0.01 
# order: 1, benchmark
pointZeros =  numpy.array([0.00354591, 1.00749553])
covZeros = numpy.array([[ 7.37064950e-06, -4.55045562e-05], [-4.55045562e-05, 7.94059221e-04]])

# general 9x9 covariance matrix for innovations
covInnovations = (0.0001 * numpy.array(
[[2.02636925, 0.83602418, -0.08710778, 0.21156024, -0.61493991, -0.1539649, 2.06356837, -9.89460925,  0.29285218],
[0.836024184, 2.91381397,  2.18979084, -0.0140301215, -0.706718228, 0.0635922479, 0.903729142, -20.3853645, 0.165924589],
[-0.0871077810, 2.18979084, 6.19334105, -0.230321489, -1.98654574, -0.0128458741, -0.0714055471, -30.2329624, 0.147309832],
[ 0.21156024, -0.01403012, -0.23032149,  0.11317097,  0.52652385, -0.02531461, 0.24972601, 5.38924994, 0.32974817],
[-0.61493991, -0.70671823, -1.98654574, 0.52652385, 39.19824138, 2.5964039, -1.174323, 88.48988427, 3.31749479],
[-0.1539649, 0.06359225, -0.01284587, -0.02531461, 2.5964039, 1.93590427, -0.84471023, -4.62833743, -0.02334074],
[2.06356837, 0.90372914, -0.07140555, 0.24972601, -1.174323, -0.84471023, 3.15243454, -7.53390082,  0.52056003],
[-9.89460925, -20.38536445, -30.23296244, 5.38924994, 88.48988427, -4.62833743, -7.53390082, 1165.88359086, 18.04011202],
[0.29285218, 0.16592459, 0.14730983, 0.32974817, 3.31749479, -0.02334074, 0.52056003, 18.04011202, 2.99990501]]))

# This is the main simulation function
# T = time horizon
def sim(initialV, initialH, initialR, initialL, T):
    
     # simulate 3d array corresponding to innovation terms
    noise = numpy.random.multivariate_normal(numpy.zeros(9), covInnovations, (T, NSIMS))
    
    # split it into components corresponding to simulated series
    noiseUSA = noise[:, :, 0] # USA stock returns Z_Q
    noiseDev = noise[:, :, 1] # international developed stock returns Z_I
    noiseEM = noise[:, :, 2] # emerging stock returns Z_E
    noiseBAA = noise[:, :, 3] # corporate bond returns Z_B
    noiseZeros = noise[:, :, 8] # Zero-coupon Treasury bond returns Z_A
    
    noiseNVol = noise[:, :, 4] # transformed volatility W_V
    noiseRates = noise[:, :, 5] # corporate bond rates W_R
    noiseMeasure = noise[:, :, 6] # the new valuation measure W_H
    noiseLSpreads = noise[:, :, 7] # the log spreads of logs W_S
    
    # simulate these regression coefficients from objective Bayes posterior
    coeffUS = numpy.random.multivariate_normal(pointUS, covUS, NSIMS)
    coeffDev = numpy.random.multivariate_normal(pointDev, covDev, NSIMS)
    coeffBAA = numpy.random.multivariate_normal(pointBAA, covBAA, NSIMS)
    coeffZeros = numpy.random.multivariate_normal(pointZeros, covZeros, NSIMS)
    coeffLSpreads = numpy.random.multivariate_normal(pointLSpreads, covLSpreads, NSIMS)
    coeffNVol = numpy.random.multivariate_normal(pointNVol, covNVol, NSIMS)
    coeffRates = numpy.random.multivariate_normal(pointRates, covRates, NSIMS)
    coeffEM = numpy.random.multivariate_normal(pointEM, covEM, NSIMS)
    coeffMeasure = numpy.random.multivariate_normal(pointMeasure, covMeasure, NSIMS)
    
    # now initialize the 2d arrays corresponding to simulated series
    simRetUSA = numpy.zeros((T, NSIMS))
    simRetIntl = numpy.zeros((T, NSIMS))
    simRetEm = numpy.zeros((T, NSIMS))
    simRetBonds = numpy.zeros((T, NSIMS))
    simRetLong = numpy.zeros((T, NSIMS))
    
    simNVol = numpy.zeros((T+1, NSIMS))
    simLRates = numpy.zeros((T+1, NSIMS))
    simLSpreads = numpy.zeros((T+1, NSIMS))
    simMeasure = numpy.zeros((T+1, NSIMS))
        
    # initialize some simulated series given initial conditions
    initialNVol = scipy.stats.yeojohnson(numpy.log(initialV), -0.5)
    simNVol[0] =  initialNVol * numpy.ones(NSIMS)
    simLRates[0] = numpy.log(initialR) * numpy.ones(NSIMS)
    simMeasure[0] = initialH * numpy.ones(NSIMS)
    initialLSpread = numpy.log(numpy.log(initialR) - numpy.log(initialL))
    simLSpreads[0] = initialLSpread * numpy.ones(NSIMS)
    
    # now comes the simulation itself!
    
    # Replace everywhere numbers with coefficients from Bayesian simulations
    
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simNVol[t + 1] = 0.8569 * numpy.ones(NSIMS) + (1 - 0.3824) * simNVol[t] + noiseVol[t]
        
    # take exponents to get volatility
    simVol = numpy.exp(YJinv(simNVol))
    
    # simulate log rates as heteroscedastic random walk
    for t in range(T):
        simLRates[t + 1] = 0.0708 + (1 - 0.0441) * simLRates[t] + noiseRates[t] * simVol[t + 1] 
        
    # take exponents to get rates
    simRates = numpy.exp(simLRates)
    
    # simulate the valuation measure as autoregression with stochastic volatility
    for t in range(T):
        simMeasure[t + 1] = 0.1699 + 0.8262 * simMeasure[t] - 0.0129 * simVol[t + 1] + simVol[t + 1] * noiseMeasure[t]

    # simulate the log spread as autoregression
    for t in range(T):
        simLSpreads[t + 1] = (1 - 0.1873) * simLSpreads[t] - 0.1980 * numpy.ones(NSIMS) + noiseSpreads[t]
        
    # simulate the long-term Treasury rate
    # We rewrite the equation S(t) = ln (R(t)) - ln (L(t))
    # as L(t) = exp(ln R(t) - S(t))
    simLong = numpy.exp(simLRates - numpy.exp(simLSpreads))
    simSpread = simRates - simLong # and simulated spreads
    
    # simulate arithmetic stock returns 
    for t in range(T):
        # three series of stock returns
        simRetUSA[t] = numpy.exp(0.2826 * numpy.ones(NSIMS) - 0.0110 * simVol[t+1] - 0.0609 * (simRates[t+1] - simRates[t]) - 0.1440 * simMeasure[t] - 0.0173 * simSpread[t] + simVol[t + 1] * noiseUSA[t]) - numpy.ones(NSIMS)
        simRetIntl[t] = numpy.exp(0.2111 * numpy.ones(NSIMS) - 0.0196 * simVol[t+1] - 0.0316 * (simRates[t+1] - simRates[t]) - 0.0641 * simMeasure[t] + 0.0387 * simSpread[t] + simVol[t+1] * noiseIntl[t]) - numpy.ones(NSIMS)
        simRetEm[t] = numpy.exp(0.0544 * numpy.ones(NSIMS) - 0.0233 * simVol[t+1] - 0.0873 * (simRates[t+1] - simRates[t]) + 0.1155 * simMeasure[t] + 0.1051 * simSpread[t] + simVol[t + 1] * noiseEm[t]) - numpy.ones(NSIMS)
        
        # two series of bond returns
        simRetBonds[t] = 0.01 * simRates[t] + numpy.exp(- 0.0596 * (simRates[t+1] - simRates[t]) + simVol[t + 1] * noiseBonds[t]) - numpy.ones(NSIMS)
        simRetLong[t] = ((numpy.ones(NSIMS) + 0.01 * simLong[t])**10)*((numpy.ones(NSIMS) + 0.01 * simLong[t+1])**(-9)) - numpy.ones(NSIMS)
         # change the line above   
    return [simRetUSA, simRetIntl, simRetEm, simRetLong, simRetBonds]
