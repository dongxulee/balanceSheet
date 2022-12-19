import mesa
import numpy as np 
import pandas as pd 

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
        
    def updateTrustMatrix(self, target):
        # update trust matrix
        self.model.trustMatrix_request[self.unique_id, target] += 1
        self.model.trustMatrix_accepted[self.unique_id, target] += 1
        self.model.trustMatrix = (self.model.trustMatrix_request + self.model.concentrationParameter) / (
        self.model.trustMatrix_accepted + self.model.concentrationParameter.sum(axis=1).reshape((self.model.num_agents, 1)))
    
    # add time decay of trust matrix
    def timeDecayOfTrustMatrix(self):
        pass
    
    def borrowRequest(self):
        if self.leverage < self.model.targetLeverageRatio:
            # randomly choose a bank to borrow from the trust matrix
            target = np.random.choice(self.model.num_agents, p=self.model.trustMatrix[self.unique_id])
            # choose a borrowing amount equal to the equity capital
            amount = self.equity
            # bring out the target bank and let him decide whether to lend
            self.other_agent = self.model.get_agent_by_id(target)
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
            if self.portfolio/2.0 > amount:
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
            self.updateBalanceSheet()
            self.borrowRequest()
            self.lendDecision()
            self.updateBalanceSheet()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, targetLeverageRatio):
        # read in banks equity capital  
        banksData = pd.read_csv(banksFile)
        N = banksData.shape[0]
        self.N = N
        self.targetLeverageRatio = targetLeverageRatio
        self.borrowingCollection = []
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # maybe we could introduce a time decay factor for trust latter 
        # all matrices are row index       
        self.trustMatrix_accepted = np.zeros((N,N))
        self.concentrationParameter = np.ones((N,N))
        self.trustMatrix = (self.trustMatrix_request + self.concentrationParameter) / (
                            self.trustMatrix_accepted + self.concentrationParameter.sum(axis=1).reshape((self.N,1)))
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
                                "Default": "default"})
        
    def updateTrustMatrix(self):
        for borrower, lender, _ in self.borrowingCollection:
            # update trust matrix with time decay 
            self.trustMatrix_accepted[borrower, lender] += 1
            self.trustMatrix = (self.trustMatrix_accepted + self.concentrationParameter)
            self.trustMatrix = self.trustMatrix / self.trustMatrix.sum(axis=1).reshape((self.N,1))
            # add time decay of trust matrix
            self.concentrationParameter = self.trustMatrix * 0.999
