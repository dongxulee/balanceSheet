from bankingSystemWithInterestAndDefault import * 
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import warnings 
warnings.filterwarnings('ignore')
import code


# simulation and data collection 
model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", 
                 targetLeverageRatio = 11.0, 
                 num_borrowing=10, 
                 num_banks=100, 
                 alpha = 0.99, 
                 beta = 0.99, 
                 fedRate = 0.0, 
                 portfolioReturnRate = 0.0) 
                 
#code.interact(local=locals())


simulationSteps = 2
for i in tqdm(range(simulationSteps)):
    model.step()

agent_data = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()