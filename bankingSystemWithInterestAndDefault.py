import mesa
import numpy as np
import pandas as pd
from eisenbergNoe import eisenbergNoe
import code

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets
        self.portfolio = 0.        # initialize when creating the bank, change in borrowing and lending
        self.lending = 0.          # update in updateBlanceSheet()
        # liabilities
        self.borrowing = 0.        # update in updateBlanceSheet()
        # equity
        self.equity = 0.           # initialize when creating the bank, update in updateBlanceSheet()
        # leverage ratio
        self.leverage = 0.         # update in updateBlanceSheet()
        # if a bank is solvent
        self.default = False      # change at clearingDebt()
    
    def updateBlanceSheet(self):
        self.equity = self.portfolio + self.lending - self.borrowing
        self.leverage = (self.portfolio + self.lending) / self.equity
        
    def borrowRequest(self):
        for _ in range(self.model.num_borrowing):
            if self.leverage < self.model.targetLeverageRatio:
                # randomly choose a bank to borrow from the trust matrix
                target = np.random.choice(self.model.N, p=self.model.trustMatrix[self.unique_id])
                # choose a borrowing amount equal to the equity capital
                amount = self.equity
                # bring out the target bank and let him decide whether to lend
                other_agent = self.model.schedule.agents[target]
                # if the lending decision is made, update the balance sheet
                if other_agent.lendDecision(self, amount):
                    self.model.L[self.unique_id, target] += amount
                    self.portfolio += amount
                    self.model.e[self.unique_id] = self.portfolio
                    self.borrowing += amount
                    self.updateBlanceSheet()
                    self.model.borrowingCollection.append([self.unique_id, target, amount])
                
    # reinforcement learning update later. 
    def lendDecision(self, borrowingBank, amount):
        # collect borrowing banks information, in this version, if the banks have enough liquidity, they will lend 
        # borrowingBank's information could be access through borrowingBank 
        if self.portfolio/2.0 > amount and np.random.rand() < 0.5:
            self.portfolio -= amount
            self.model.e[self.unique_id] = self.portfolio
            self.lending += amount
            # asset and equity amount remain unchanged, leverage ratio also remains unchanged
            return True
        else:
            return False
            
    def returnOnPortfolio(self):
        self.portfolio = self.portfolio * (1+self.model.portfolioReturnRate)
        self.updateBlanceSheet()
    
    def reset(self):
        self.portfolio = self.model.e[self.unique_id][0]
        self.lending = 0.    
        # liabilities
        self.borrowing = 0.      
        # equity
        self.equity = self.portfolio   
        # leverage ratio
        self.leverage = 1.0
        if self.portfolio < 0.:
            self.default = True
    
    def step(self):
        if not self.default:
            self.returnOnPortfolio()
            self.borrowRequest()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, 
                 targetLeverageRatio, 
                 num_borrowing, 
                 num_banks, 
                 alpha = 0.99,
                 beta = 0.99,
                 concentrationParameter = None, 
                 fedRate = 0., 
                 portfolioReturnRate = 0.):
        
        # interest rate
        self.fedRate = fedRate
        # portfolio return rate
        self.portfolioReturnRate = portfolioReturnRate
        # asset recovery rate 
        self.alpha = alpha
        # interbank loan recovery rate
        self.beta = beta
        
        # read in banks equity capital
        banksData = pd.read_csv(banksFile).iloc[:num_banks,:]
        self.banks = banksData["bank"]
        self.N = num_banks
        self.targetLeverageRatio = targetLeverageRatio
        self.num_borrowing = num_borrowing
        self.borrowingCollection = []
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # we also introduce a time decay factor for trust       
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.N,self.N))
            np.fill_diagonal(self.concentrationParameter, 0.)
        else:
            self.concentrationParameter = concentrationParameter
        self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
        # liability matrix 
        self.L = np.zeros((self.N,self.N))
        # asset matrix
        self.e = np.zeros((self.N,1))
        # create a schedule for banks
        self.schedule = mesa.time.RandomActivation(self)
    
        # create banks and put them in schedule
        for i in range(self.N):
            a = Bank(i, self)
            a.portfolio = banksData["equity"][i]
            a.equity = banksData["equity"][i]
            self.schedule.add(a)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trust Matrix": "trustMatrix", 
                             "Liability Matrix": "L",
                             "Asset Matrix": "e"},
            agent_reporters={"PortfolioValue": "portfolio",
                                "Lending": "lending",
                                "Borrowing": "borrowing", 
                                "Equity": "equity",
                                "Default": "default",
                                "Leverage": "leverage"})
        
    def updateTrustMatrix(self):
        if len(self.borrowingCollection) > 0:
            borrowingCollection = np.array(self.borrowingCollection)
            borrowerIndex = borrowingCollection[:,0].astype(int)
            lenderIndex = borrowingCollection[:,1].astype(int)
            # add time decay of concentration parameter
            self.concentrationParameter = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True) * (self.N - 1)
            # update trust matrix 
            self.concentrationParameter[borrowerIndex, lenderIndex] += 1.
            self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
            # clean the borrowing collection
            self.borrowingCollection = []
            
    def clearingDebt(self):
        # Returns the new portfolio value after clearing debt
        _, e, insolventBanks = eisenbergNoe(self.L, self.e, self.alpha, self.beta)
        self.e = e
        self.L = np.zeros((self.N,self.N))
        if len(insolventBanks) > 0:
            self.concentrationParameter[:,insolventBanks] = 0
        for agent in self.schedule.agents:
            agent.reset()

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.updateTrustMatrix()
        self.clearingDebt()
