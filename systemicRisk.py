from bankingSystem import * 
from helperFunctions import *
import numpy as np
import warnings 
import multiprocessing
warnings.filterwarnings('ignore')

con = np.load("concentrationParams.npy")

################################################################################## Base Model  
def run(iRun):
    np.random.seed(iRun+1000)
    # simulation and data collection
    simulationSteps = 500
    cMatrix = np.ones((100,100))*0.8
    np.fill_diagonal(cMatrix, 1)
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
                    depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
                    num_borrowing= 0,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.5,                                              # portfolio recovery rate                           
                    beta = 0.9,                                               # interbank loan recovery rate
                    fedRate = 0.04,                                            # interest rate on borrowing   
                    portfolioReturnRate = 0.10,          
                    # return rate on portfolio
                    returnVolatiliy = 0.18,
                    returnCorrelation = cMatrix,
                    liquidityShockNum = 10,                                    # number of liquidity shocks per step      
                    shockSize = 0.02,                                          # size of the shock
                    shockDuration = [300, 300]) # duration of the shock
                    
    model.datacollector.collect(model)
    for _ in range(simulationSteps):
        model.simulate()
        
    agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()
    return model_data, agent_data

def multiRun(numberOfruns):
    # running the simulation in parallel
    cpuNum = 48
    results = []
    batchNum = 0
    while numberOfruns > 0:
        print(numberOfruns)
        if numberOfruns > cpuNum:
            numberOfruns = numberOfruns - cpuNum
            with multiprocessing.Pool() as pool:
                # run the function in parallel on the input values
                results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
            batchNum = batchNum + 1
        else:
            with multiprocessing.Pool() as pool:
                results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
            numberOfruns = 0
    return results

def defaultNumber(results):
    defaultCollection = []
    for iter in range(len(results)):
        m, a = results[iter]
        defaultCollection.append(a.xs(500, level="Step")["Default"].sum())
    return defaultCollection

def defaultBank(results):
    defaultCollection = []
    for iter in range(len(results)):
        m, a = results[iter]
        defaultCollection.append(a.xs(500, level="Step")["Default"].values)
    return np.array(defaultCollection)

defaultCollection = []
defaultBanks = []
results = multiRun(2000)
defaultCollection.append(defaultNumber(results))
defaultBanks.append(defaultBank(results))
print("Base Model Done")
# ################################################################################## High Reserve Model  
# def run(iRun):
#     np.random.seed(iRun+1000)
#     # simulation and data collection
#     simulationSteps = 500
#     cMatrix = np.ones((100,100))*0.8
#     np.fill_diagonal(cMatrix, 1)
#     model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
#                     leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
#                     depositReserve = 0.4,                                     # capital reserve as a ratio of portfolio value
#                     num_borrowing= 10,                                        # number of borrowing request per bank per step
#                     sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
#                     num_banks=100,                                            # number of banks in the system 
#                     alpha = 0.5,                                              # portfolio recovery rate                           
#                     beta = 0.9,                                               # interbank loan recovery rate
#                     fedRate = 0.04,                                            # interest rate on borrowing   
#                     portfolioReturnRate = 0.10,          
#                     # return rate on portfolio
#                     returnVolatiliy = 0.18,
#                     returnCorrelation = cMatrix,
#                     liquidityShockNum = 10,                                    # number of liquidity shocks per step      
#                     shockSize = 0.02,                                          # size of the shock
#                     shockDuration = [300, 300]) # duration of the shock
                    
#     model.datacollector.collect(model)
#     for _ in range(simulationSteps):
#         model.simulate()
        
#     agent_data = model.datacollector.get_agent_vars_dataframe()
#     model_data = model.datacollector.get_model_vars_dataframe()
#     return model_data, agent_data

# def multiRun(numberOfruns):
#     # running the simulation in parallel
#     cpuNum = 48
#     results = []
#     batchNum = 0
#     while numberOfruns > 0:
#         if numberOfruns > cpuNum:
#             numberOfruns = numberOfruns - cpuNum
#             with multiprocessing.Pool() as pool:
#                 # run the function in parallel on the input values
#                 results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
#             batchNum = batchNum + 1
#         else:
#             with multiprocessing.Pool() as pool:
#                 results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
#             numberOfruns = 0
#     return results

# results = multiRun(1000)
# defaultCollection.append(defaultNumber(results))
# defaultBanks.append(defaultBank(results))
# print("High Reserve Model Done")
# ################################################################################## Low Leverage Model  
# def run(iRun):
#     np.random.seed(iRun+1000)
#     # simulation and data collection
#     simulationSteps = 500
#     cMatrix = np.ones((100,100))*0.8
#     np.fill_diagonal(cMatrix, 1)
#     model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
#                     leverageRatio = 15.0,                                     # leverage ratio upper bound for all banks
#                     depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
#                     num_borrowing= 10,                                        # number of borrowing request per bank per step
#                     sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
#                     num_banks=100,                                            # number of banks in the system 
#                     alpha = 0.5,                                              # portfolio recovery rate                           
#                     beta = 0.9,                                               # interbank loan recovery rate
#                     fedRate = 0.04,                                            # interest rate on borrowing   
#                     portfolioReturnRate = 0.10,          
#                     # return rate on portfolio
#                     returnVolatiliy = 0.18,
#                     returnCorrelation = cMatrix,
#                     liquidityShockNum = 10,                                    # number of liquidity shocks per step      
#                     shockSize = 0.02,                                          # size of the shock
#                     shockDuration = [300, 300]) # duration of the shock
                    
#     model.datacollector.collect(model)
#     for _ in range(simulationSteps):
#         model.simulate()
        
#     agent_data = model.datacollector.get_agent_vars_dataframe()
#     model_data = model.datacollector.get_model_vars_dataframe()
#     return model_data, agent_data

# def multiRun(numberOfruns):
#     # running the simulation in parallel
#     cpuNum = 48
#     results = []
#     batchNum = 0
#     while numberOfruns > 0:
#         if numberOfruns > cpuNum:
#             numberOfruns = numberOfruns - cpuNum
#             with multiprocessing.Pool() as pool:
#                 # run the function in parallel on the input values
#                 results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
#             batchNum = batchNum + 1
#         else:
#             with multiprocessing.Pool() as pool:
#                 results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
#             numberOfruns = 0
#     return results

# results = multiRun(1000)
# defaultCollection.append(defaultNumber(results))
# defaultBanks.append(defaultBank(results))
# print("Low Leverage Model Done")
# ################################################################################## Low correlation Model  
# def run(iRun):
#     np.random.seed(iRun+1000)
#     # simulation and data collection
#     simulationSteps = 500
#     cMatrix = np.ones((100,100))*0.4
#     np.fill_diagonal(cMatrix, 1)
#     model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
#                     leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
#                     depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
#                     num_borrowing= 10,                                        # number of borrowing request per bank per step
#                     sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
#                     num_banks=100,                                            # number of banks in the system 
#                     alpha = 0.5,                                              # portfolio recovery rate                           
#                     beta = 0.9,                                               # interbank loan recovery rate
#                     fedRate = 0.04,                                            # interest rate on borrowing   
#                     portfolioReturnRate = 0.10,          
#                     # return rate on portfolio
#                     returnVolatiliy = 0.18,
#                     returnCorrelation = cMatrix,
#                     liquidityShockNum = 10,                                    # number of liquidity shocks per step      
#                     shockSize = 0.02,                                          # size of the shock
#                     shockDuration = [300, 300]) # duration of the shock
                    
#     model.datacollector.collect(model)
#     for _ in range(simulationSteps):
#         model.simulate()
        
#     agent_data = model.datacollector.get_agent_vars_dataframe()
#     model_data = model.datacollector.get_model_vars_dataframe()
#     return model_data, agent_data

# def multiRun(numberOfruns):
#     # running the simulation in parallel
#     cpuNum = 48
#     results = []
#     batchNum = 0
#     while numberOfruns > 0:
#         if numberOfruns > cpuNum:
#             numberOfruns = numberOfruns - cpuNum
#             with multiprocessing.Pool() as pool:
#                 # run the function in parallel on the input values
#                 results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
#             batchNum = batchNum + 1
#         else:
#             with multiprocessing.Pool() as pool:
#                 results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
#             numberOfruns = 0
#     return results

# results = multiRun(1000)
# defaultCollection.append(defaultNumber(results))
# defaultBanks.append(defaultBank(results))
# print("Low correlation Model Done")
# ################################################################################## No correlation Model  
# def run(iRun):
#     np.random.seed(iRun+1000)
#     # simulation and data collection
#     simulationSteps = 500
#     cMatrix = np.ones((100,100))*0.0
#     np.fill_diagonal(cMatrix, 1)
#     model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
#                     leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
#                     depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
#                     num_borrowing= 10,                                        # number of borrowing request per bank per step
#                     sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
#                     num_banks=100,                                            # number of banks in the system 
#                     alpha = 0.5,                                              # portfolio recovery rate                           
#                     beta = 0.9,                                               # interbank loan recovery rate
#                     fedRate = 0.04,                                            # interest rate on borrowing   
#                     portfolioReturnRate = 0.10,          
#                     # return rate on portfolio
#                     returnVolatiliy = 0.18,
#                     returnCorrelation = cMatrix,
#                     liquidityShockNum = 10,                                    # number of liquidity shocks per step      
#                     shockSize = 0.02,                                          # size of the shock
#                     shockDuration = [300, 300]) # duration of the shock
                    
#     model.datacollector.collect(model)
#     for _ in range(simulationSteps):
#         model.simulate()
        
#     agent_data = model.datacollector.get_agent_vars_dataframe()
#     model_data = model.datacollector.get_model_vars_dataframe()
#     return model_data, agent_data

# def multiRun(numberOfruns):
#     # running the simulation in parallel
#     cpuNum = 48
#     results = []
#     batchNum = 0
#     while numberOfruns > 0:
#         if numberOfruns > cpuNum:
#             numberOfruns = numberOfruns - cpuNum
#             with multiprocessing.Pool() as pool:
#                 # run the function in parallel on the input values
#                 results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
#             batchNum = batchNum + 1
#         else:
#             with multiprocessing.Pool() as pool:
#                 results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
#             numberOfruns = 0
#     return results

# results = multiRun(1000)
# defaultCollection.append(defaultNumber(results))
# defaultBanks.append(defaultBank(results))
# print("No correlation Model Done")
# ################################################################################## Low borrow Model  
# def run(iRun):
#     np.random.seed(iRun+1000)
#     # simulation and data collection
#     simulationSteps = 500
#     cMatrix = np.ones((100,100))*0.8
#     np.fill_diagonal(cMatrix, 1)
#     model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
#                     leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
#                     depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
#                     num_borrowing= 5,                                        # number of borrowing request per bank per step
#                     sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
#                     num_banks=100,                                            # number of banks in the system 
#                     alpha = 0.5,                                              # portfolio recovery rate                           
#                     beta = 0.9,                                               # interbank loan recovery rate
#                     fedRate = 0.04,                                            # interest rate on borrowing   
#                     portfolioReturnRate = 0.10,          
#                     # return rate on portfolio
#                     returnVolatiliy = 0.18,
#                     returnCorrelation = cMatrix,
#                     liquidityShockNum = 10,                                    # number of liquidity shocks per step      
#                     shockSize = 0.02,                                          # size of the shock
#                     shockDuration = [300, 300]) # duration of the shock
                    
#     model.datacollector.collect(model)
#     for _ in range(simulationSteps):
#         model.simulate()
        
#     agent_data = model.datacollector.get_agent_vars_dataframe()
#     model_data = model.datacollector.get_model_vars_dataframe()
#     return model_data, agent_data

# def multiRun(numberOfruns):
#     # running the simulation in parallel
#     cpuNum = 48
#     results = []
#     batchNum = 0
#     while numberOfruns > 0:
#         if numberOfruns > cpuNum:
#             numberOfruns = numberOfruns - cpuNum
#             with multiprocessing.Pool() as pool:
#                 # run the function in parallel on the input values
#                 results = results + pool.map(run, range(batchNum*cpuNum, (batchNum+1)*cpuNum))
#             batchNum = batchNum + 1
#         else:
#             with multiprocessing.Pool() as pool:
#                 results = results + pool.map(run, range(batchNum*cpuNum, batchNum*cpuNum + numberOfruns))
#             numberOfruns = 0
#     return results

# results = multiRun(1000)
# defaultCollection.append(defaultNumber(results))
# defaultBanks.append(defaultBank(results))
# print("Low borrow Model Done")


defaultCollection = np.array(defaultCollection)
np.save("defaultCollection_noBorrow.npy", defaultCollection)

defaultBanks = np.array(defaultBanks)
np.save("defaultBanks_noBorrow.npy", defaultBanks)