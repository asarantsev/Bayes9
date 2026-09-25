# Bayes9
Fully updated Bayesian simulator with 8 innovation series. 5 asset classes of US stocks, developed stocks, emerging stocks, Treasury bonds, and corporate bonds. All returns are total, nominal. Valuation measure, log spread of logs, BAA log rates, volatility. 4 factors: Measure (last day of year), volatility (annual), BAA rate, 10-year Treasury rate (average daily December). This is NOT a stationary model. All innovations are IID Gaussian, tested by Shapiro-Wilk and Jarque-Bera normality tests, and by the Ljung-Box test for 5 and 10 lags for original and absolute values of innovations. Features:

Volatility and S&P returns data thanks to Angel Piotrowski and Ian Anderson

Rate data for BAA and 10-year Treasury extended back to 1925 from 1927

Corporate bond returns data extended back to 1926 from 1973 from SBBI thanks to Smaila Amoanu

Yeo-Johnson transform of log volatility with index -0.5 modeled as autoregression with lag 1

Log risk spread of logs as autoregression with lag 1

Valuation measure modeled as autoregression with lag 1 with stochastic volatility

Log BAA rate as random walk with stochastic volatility

Treasury 10-year returns (coupon bonds) deterministic formula thanks to Aswath Damodaran

Corporate bond returns minus rates (log duration) as a function of rate change

Domestic, developed, emerging stocks vs rate change and valuation measure with stochastic volatility

TBD: Bayesian version with objective Bayes inference for regression coefficients; and maybe we can add: For international stocks, its own dividend-based valuation measure rather than rely on US-based; for Treasury bonds, use actual returns data from SBBI or FRED to compare with predicted returns from Damodaran's formula

https://my-finance.org/2026/09/25/current-simulation-version/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Previous: 

Returns of 5 asset classes of US stocks, developed stocks, emerging stocks, Treasury bonds, and corporate bonds. (All returns are geometric, total, nominal.) Valuation measure, log spread of logs, BAA log rates, Yeo-Johnson transformed log volatility. 4 factors: Measure (last day of year), volatility (annual), BAA rate, 10-year Treasury rate (average daily December).

; Randomness for 7-year zero-coupon Treasury returns; Objective Bayes inference for regression coefficients but not for standard errors or correlation matrix; Normalization of each of three Valuation and duration.

We still have to make further updates. We have a data file with all 5 assets. Python file with fitted regressions (no plots, they will be added later) and another Python file with Bayesian simulation of regression coefficients but not covariance matrix of innovations (coefficients need to be updated, together with zero-coupon bond returns). 

Next we must add the plot simulation, and the overall file which we start in place of a web app, and the classic simulation file. Also, add different models: simplified ones. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UPDATE: We renamed the previous versions old-fit.py and old-data.xlsx. We put them into the archive folder. 

We removed a Python file with Bayesian simulation. But we added more files: fit-model.py and full-data.xlsx. Here we model Treasury returns deterministically. And include Ibbotson SBBI corporate returns data, 1926-1975.

Finally, we included the file with classic (NOT Bayesian simulation) of the 5 asset classes, adapted from the repository https://github.com/asarantsev/Gaussian-Innovations/tree/main/full-model

The next and final step would be to include Bayesian version. It occurred to us we might simply use Wishart matrix-valued distribution with 100 degrees of freedom (corresponding to the maximal number of regression data points).

