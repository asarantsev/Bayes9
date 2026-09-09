# Bayes9
Fully updated Bayesian simulator with 9 regressions, 9 innovation series:

Returns of US stocks, developed stocks, emerging stocks, Treasury bonds, and corporate bonds. (All returns are geometric, total, nominal.) Valuation measure, log spread of logs, BAA log rates, Yeo-Johnson transformed log volatility. 

These 9 are out of 5 asset classes and 4 factors: Measure (last day of year), volatility (annual), BAA rate, 10-year Treasury rate (average daily December).

Yeo-Johnson transform of log volatility with index -0.5; Randomness for 7-year zero-coupon Treasury returns; Objective Bayes inference for regression coefficients but not for standard errors or correlation matrix; Normalization of each of three Valuation and duration.
