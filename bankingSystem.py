import mesa
import numpy as np 
import pandas as pd 

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets  
        self.portfolio = 0
        self.lendingTarget = []
        self.lendingAmount = []
        self.lending = np.sum(self.lendingAmount)
        # liabilities
        self.borrowingTarget = []
        self.borrowingAmount = []
        self.borrowing = np.sum(self.borrowingAmount)
        # equity 
        self.equity = 0
        # if a bank is solvent
        self.default = False
    
    def borrowRequest(self):
        if self.borrowing

    def lendDecision(self):
        # based on the borrowing request, decide whether to lend
        pass
    
    def updateBalanceSheet(self):
        pass

    def step(self):
        if not self.default:
            self.updateBalanceSheet()
            self.borrowRequest()
            self.lendDecision()
            self.updateBalanceSheet()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile):
        # read in banks equity capital  
        banksData = pd.read_csv(banksFile)
        N = banksData.shape[0]
        self.num_agents = N
        self.trustMatrix = np.random.rand(N,N)
        self.schedule = mesa.time.RandomActivation(self)
        
        # create banks and put them in schedule
        for i in range(self.num_agents):
            a = Bank(i, self)
            self.schedule.add(a)
            a.portfolio = banksData["equity"][i]
            a.equity = banksData["equity"][i]
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trust Matrix": "trustMatrix",},
            agent_reporters={"Portfolio Value": "portfolio",
                                "Lending": "lending",
                                "Borrowing": "borrowing", 
                                "Equity": "equity",
                                "Default": "default"} 
     )
