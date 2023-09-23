from bankingSystem import * 
from helperFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings 
warnings.filterwarnings('ignore')

# start from here
bankFile = "df_detail.csv"
params = {"banksFile" : bankFile, # csv file used to initialize the bank agents
                 "leverageRatio": 10.0, # leverage ratio upper bound for all banks
                 "cashReserve": 0.05, # capital reserve as a ratio of deposit
                 "num_borrowing": 10, # number of borrowing request per bank per step
                 "sizeOfBorrowing": 1.0, # size of borrowing as a ratio of equity capital
                 "concentrationParameter": None, #np.load("concentrationParams.npy"), # concentration parameter for the dirichlet distribution
                 "num_banks": 50, # number of banks in the system 
                 "alpha" : 0.5,    # portfolio recovery rate          
                 "beta" : 0.9,     # interbank loan recovery rate
                 "fedRate" : 0.04, # interest rate on borrowing   
                 "portfolioReturnRate" : 0.10, 
                 "returnVolatiliy" : 0.18,
                 "returnCorrelation" : 0.9,
                 "shockSize" : 10,       # size of the shock
                 "shockDuration":[-1,-1] # time of the shock, [-1,-1] sugguests no shock
                 } 

numberOfRun = 1
simulationSteps = 500
rCollection = multiRun(numberOfRun, params)