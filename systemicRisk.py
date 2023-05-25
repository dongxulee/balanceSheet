from bankingSystem import * 
from helperFunctions import *
import numpy as np
import warnings 
import copy
warnings.filterwarnings('ignore')

# start from here
bankFile = "balanceSheetAnalysis/banksData_2022.csv"
params = {"banksFile" : bankFile, # csv file used to initialize the bank agents
                 "leverageRatio": 10.0, # leverage ratio upper bound for all banks
                 "depositReserve": 0.2, # capital reserve as a ratio of deposit
                 "num_borrowing": 5, # number of borrowing request per bank per step
                 "sizeOfBorrowing": 1.0, # size of borrowing as a ratio of equity capital
                 "concentrationParameter": np.load("concentrationParams.npy"), # concentration parameter for the dirichlet distribution
                 "num_banks": 100, # number of banks in the system 
                 "alpha" : 0.5,    # portfolio recovery rate          
                 "beta" : 0.9,     # interbank loan recovery rate
                 "fedRate" : 0.04, # interest rate on borrowing   
                 "portfolioReturnRate" : 0.10, 
                 "returnVolatiliy" : 0.18,
                 "returnCorrelation" : 0.8,
                 "liquidityShockNum" : 0,  # number of liquidity shocks per step (not correlated shocks)
                 "shockSize" : 5,       # size of the shock
                 "shockDuration":[300,300] # time of the shock, [-1,-1] sugguests no shock
                 } 
numberOfRuns = 1000

# collect the default statistics
def defaultNumber(results):
    defaultCollection = []
    for iter in range(len(results)):
        a,m = results[iter]
        defaultCollection.append(a.xs(500, level="Step")["Default"].sum())
    return defaultCollection

def defaultBank(results):
    defaultCollection = []
    for iter in range(len(results)):
        a,m = results[iter]
        defaultCollection.append(a.xs(500, level="Step")["Default"].values)
    return np.array(defaultCollection)

def runAndWriteToFile(numberOfRuns, params, fileName):
    defaultCollection = []
    defaultBanks = []
    results = multiRun(numberOfRuns, params)
    defaultCollection.append(defaultNumber(results))
    defaultCollection = np.array(defaultCollection)
    defaultBanks.append(defaultBank(results))
    defaultBanks = np.array(defaultBanks)
    
    np.save("defaultSimulation/defaultCollection_" + fileName + ".npy", defaultCollection)
    np.save("defaultSimulation/defaultBanks_" + fileName + ".npy", defaultBanks)
    
################################################################################## Base Model  
params1 = copy.deepcopy(params)
runAndWriteToFile(numberOfRuns, params1, "baseModel")
print("Base Model Done")
################################################################################## High Reserve Model  
params2 = copy.deepcopy(params)
params2["depositReserve"] = 0.4
runAndWriteToFile(numberOfRuns, params2, "highReserve")
print("High Reserve Model Done")
################################################################################## Low Leverage Model  
params3 = copy.deepcopy(params)
params3["leverageRatio"] = 15.0
runAndWriteToFile(numberOfRuns, params3, "lowLeverage") 
print("Low Leverage Model Done")
################################################################################## Low correlation Model  
params4 = copy.deepcopy(params)
params4["returnCorrelation"] = 0.4
runAndWriteToFile(numberOfRuns, params4, "lowCorrelation")
print("Low correlation Model Done")
################################################################################## No correlation Model  
params5 = copy.deepcopy(params)
params5["returnCorrelation"] = 0.0
runAndWriteToFile(numberOfRuns, params5, "noCorrelation")
print("No correlation Model Done")
################################################################################## Low borrow Model
params6 = copy.deepcopy(params)
params6["num_borrowing"] = 2 
runAndWriteToFile(numberOfRuns, params6, "lowBorrow")
print("Low borrow Model Done")
################################################################################## No borrow Model
params7 = copy.deepcopy(params)
params7["num_borrowing"] = 0
runAndWriteToFile(numberOfRuns, params7, "noBorrow")