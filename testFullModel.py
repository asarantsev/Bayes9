# This package allows us to simulate portfolios and plot Monte Carlo simulations

# USA stocks measured by Standard & Poor
# developed ex-USA stocks, measured by MSCI EAFE 88% + MSCI Canada 12%
# emerging markets stocks, measured by MSCI EM
# USA long-term Treasuries, measured by 10-year zero-coupon Treasury bonds
# USA investment-grade corporate bonds, measured by MSCI index

# change input parameters indicated below
# and files simFullModel and appFullModel must be in the same folder

# it saves PNG and PDF with graphs of portfolios and summary statistics

from simFullModel import sim
from appFullModel import output

T = 5 # time horizon
initialW = 1000000 # initial wealth
initialFlow = -40000 # initial flow
growthFlow = 0.03 # arithmetic annual flow growth
initialV = 10 # initial annual volatility
initialR = 4 # initial BAA rate
initialH = 0 # initial S&P valuation measure
initialL = 2.5 # initial 10-year Treasury rate
startBond = 0.3 # initial stock/bond split = 70/30
endBond = 0.1 # terminal stock/bond split = 90/10
intl = 0.6 # international share in stock part
em = 0.5 # emerging share in international stock part
treas = 0.4 # 10-year Treasuries share in bond part

simulation = sim(initialV, initialH, initialR, initialL, T)
output(simulation, initialW, initialFlow, growthFlow, T, startBond, endBond, intl, em, treas, initialV, initialH, initialR, initialL, 'output-graph')