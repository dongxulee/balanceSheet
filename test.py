from bankingSystemRL import * 
from helperFunctions import *
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')


# simulation and data collection
simulationSteps = 500
gradientSteps = 100
def R_tau(model_data):
    value = (model_data['Asset Matrix'][simulationSteps] +
    model_data['Liability Matrix'][simulationSteps].T.sum(axis = 1, keepdims = True) - model_data['Liability Matrix'][simulationSteps].sum(axis = 1, keepdims=True)) - (model_data['Asset Matrix'][0] +
    model_data['Liability Matrix'][0].T.sum(axis = 1, keepdims = True) - model_data['Liability Matrix'][0].sum(axis = 1, keepdims=True))
    value[value <= 0] = -np.log(1-value[value <= 0])
    value[value > 0] = np.log(1+value[value > 0])
    return value

stepSize = 0.001
r_collection = []
w_collection = []
w = np.zeros(6)
for _ in tqdm(range(gradientSteps)):
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 11.0,                                     # leverage ratio upper bound for all banks
                    capitalReserve = 0.0,                                     # capital reserve as a ratio of portfolio value
                    num_borrowing= 20,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.0,                                              # portfolio recovery rate                           
                    beta = 0.5,                                               # interbank loan recovery rate
                    fedRate = 0.04/252,                                       # interest rate on borrowing   
                    portfolioReturnRate = 0,                                  # return rate on portfolio
                    liquidityShockNum = 3,                                    # number of liquidity shocks per step      
                    shockSize = 0.,                                           # size of the shock
                    shockDuration =  [simulationSteps // 10 * 6, simulationSteps // 10 * 7], # duration of the shock
                    w = w)                                                    # policy function weights
                    
    model.datacollector.collect(model)
    for i in tqdm(range(simulationSteps)):
        model.simulate()

    #agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()

    r = R_tau(model_data)
    gradient = np.zeros(w.size)
    for j in range(r.size):
        gradient += model.schedule.agents[j].gradient * r[j][0]
    gradient = gradient / r.size
    w = w + stepSize*gradient
    w_collection.append(w)
    r_collection.append(r[0])
    
np.save('w_collection', w_collection)
np.save('r_collection', r_collection)