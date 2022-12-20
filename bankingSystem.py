import mesa
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets  
        self.portfolio = 1
        self.lendingTarget = []
        self.lendingAmount = []
        self.lending = np.sum(self.lendingAmount)
        # liabilities
        self.borrowingTarget = []
        self.borrowingAmount = []
        self.borrowing = np.sum(self.borrowingAmount)
        # equity 
        self.equity = 1
        # leverage ratio
        self.leverage = (self.portfolio + self.lending) / self.equity
        # if a bank is solvent
        self.default = False
    
    
    def borrowRequest(self):
        for _ in range(self.model.num_borrowing):
            if self.leverage < self.model.targetLeverageRatio:
                # randomly choose a bank to borrow from the trust matrix
                target = np.random.choice(self.model.N, p=self.model.trustMatrix[self.unique_id])
                if target != self.unique_id:
                    # choose a borrowing amount equal to the equity capital
                    amount = self.equity
                    # bring out the target bank and let him decide whether to lend
                    self.other_agent = self.model.schedule.agents[target]
                    # if the lending decision is made, update the balance sheet
                    if self.other_agent.lendDecision(self, amount):
                        self.borrowingTarget.append(self.unique_id)
                        self.borrowingAmount.append(amount)
                        self.borrowing = np.sum(self.borrowingAmount)
                        self.portfolio = self.portfolio + amount
                        self.leverage = (self.portfolio + self.lending) / self.equity
                        self.model.borrowingCollection.append((self.unique_id, target, amount))
                

    def lendDecision(self, borrowingBank, amount):
        # if the bank is solvent
        if not self.default: 
        # collect borrowing banks information, in this version, if the banks have enough liquidity, they will lend 
            if self.portfolio/2.0 > amount and np.random.rand() < 0.5:
                self.lendingTarget.append(borrowingBank.unique_id)
                self.lendingAmount.append(amount)
                self.portfolio = self.portfolio - amount
                self.lending = np.sum(self.lendingAmount)
                # asset and equity amount remain unchanged, leverage ratio also remains unchanged
                return True
            else:
                return False
            
    
    def step(self):
        if not self.default:
            self.borrowRequest()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, targetLeverageRatio, num_borrowing, num_banks):
        # read in banks equity capital  
        banksData = pd.read_csv(banksFile).iloc[:num_banks,:]
        self.banks = banksData["bank"]
        self.N = banksData.shape[0]
        self.targetLeverageRatio = targetLeverageRatio
        self.num_borrowing = num_borrowing
        self.borrowingCollection = []
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # maybe we could introduce a time decay factor for trust latter 
        # all matrices are row index       
        self.trustMatrix_accepted = np.zeros((self.N,self.N))
        self.concentrationParameter = np.ones((self.N,self.N))
        np.fill_diagonal(self.concentrationParameter, 0)
        self.trustMatrix = self.trustMatrix_accepted + self.concentrationParameter
        self.trustMatrix =self.trustMatrix / self.trustMatrix.sum(axis=1, keepdims=True)
        self.schedule = mesa.time.RandomActivation(self)
    
        # create banks and put them in schedule
        for i in range(self.N):
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
                                "Default": "default",
                                "Leverage": "leverage"})
        
    def updateTrustMatrix(self):
        for borrower, lender, _ in self.borrowingCollection:
            # update trust matrix with time decay 
            self.trustMatrix_accepted[borrower, lender] += 1
        self.trustMatrix = self.trustMatrix_accepted + self.concentrationParameter
        self.trustMatrix = self.trustMatrix / self.trustMatrix.sum(axis=1, keepdims=True)
        # add time decay of trust matrix
        self.concentrationParameter = self.trustMatrix * (1-1e-10)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.updateTrustMatrix()
