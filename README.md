# Bayes9
Fully updated Bayesian simulator with 9 regressions, 9 innovation series.

Returns of 5 asset classes of US stocks, developed stocks, emerging stocks, Treasury bonds, and corporate bonds. (All returns are geometric, total, nominal.) Valuation measure, log spread of logs, BAA log rates, Yeo-Johnson transformed log volatility. 4 factors: Measure (last day of year), volatility (annual), BAA rate, 10-year Treasury rate (average daily December).

Yeo-Johnson transform of log volatility with index -0.5; Randomness for 7-year zero-coupon Treasury returns; Objective Bayes inference for regression coefficients but not for standard errors or correlation matrix; Normalization of each of three Valuation and duration.

We still have to make further updates. We have a data file with all 5 assets. Python file with fitted regressions (no plots, they will be added later) and another Python file with Bayesian simulation of regression coefficients but not covariance matrix of innovations (coefficients need to be updated, together with zero-coupon bond returns). 

Next we must add the plot simulation, and the overall file which we start in place of a web app, and the classic simulation file. Also, add different models: simplified ones. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UPDATE: We renamed the previous versions old-fit.py and old-data.xlsx. We removed a Python file with Bayesian simulation. But we added more files.

