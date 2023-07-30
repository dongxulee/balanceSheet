import mesa
import numpy as np
import pandas as pd
from eisenbergNoe import eisenbergNoe

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets
        self.cash = 0.        # initialize when creating the bank, change in borrowing and lending
        self.sec = 0.
        self.ill = 0.
        self.lending = 0.         
        self.asset = 0.
        # liabilities
        self.borrowing = 0.       
        self.deposit = 0.        
        # equity
        self.equity = 0.           # initialize when creating the bank, update in updateBlanceSheet()
        # leverage ratio
        self.leverage = 0.        
        # if a bank is solvent
        self.default = 0      # change at clearingDebt()
    
    def updateBlanceSheet(self):
        # equity = asset - liability
        self.asset = self.cash + self.sec + self.ill + self.lending
        self.equity = self.asset - self.deposit - self.borrowing
        # leverage ratio = asset / equity
        self.leverage = self.asset / self.equity
        
    def borrowRequest(self):
        for _ in range(self.model.num_borrowing):
            if self.leverage < self.model.leverageRatio:
                # randomly choose a bank to borrow from the trust matrix
                prob = self.model.trustMatrix[self.unique_id]
                # only one bank remains solvent
                if np.isnan(prob).any():
                    break
                target = np.random.choice(self.model.N, p=prob)
                # choose a borrowing amount equal to the equity capital
                amount = self.equity * self.model.sizeOfBorrowing
                # bring out the target bank and let him decide whether to lend
                other_agent = self.model.schedule.agents[target]
                # if the lending decision is made, update the balance sheet
                if other_agent.lendDecision(self, amount):
                    self.model.L[self.unique_id, target] += amount
                    self.cash += amount
                    self.model.e_cash[self.unique_id] = self.cash
                    self.borrowing += amount
                    self.updateBlanceSheet()
                    self.model.e[self.unique_id][0] = self.asset
                    self.model.concentrationParameter[self.unique_id, target] += 1.
                
    def lendDecision(self, borrowingBank, amount):
        # collect borrowing banks information, in this version, if the banks have enough liquidity, they will lend 
        # borrowingBank's information could be access through borrowingBank 
        # maintain a cash level of 5% of the asset
        if self.cash - self.asset*self.model.cashReserve > amount:
            self.cash -= amount
            self.model.e_cash[self.unique_id] = self.cash
            self.lending += amount
            # asset and equity amount remain unchanged, leverage ratio also remains unchanged
            return True
        else:
            return False
    
    def reset(self):
        self.asset = self.model.e[self.unique_id][0]
        # if default
        if self.asset == 0:
            # assets
            self.cash = 0.        
            self.sec = 0.
            self.ill = 0.
            self.lending = 0.         
            # liabilities
            self.borrowing = 0.       
            self.deposit = 0.        
            # equity
            self.equity = 0.          
            # leverage ratio
            self.leverage = 0.        
            self.default = 1 
        else:
            # assets
            self.cash = self.model.e_cash[self.unique_id][0]      
            self.sec = self.model.e_sec[self.unique_id][0]
            self.ill = self.model.e_ill[self.unique_id][0]
            self.lending = 0.         
            # liabilities
            self.borrowing = 0.       
            self.deposit = self.model.d[self.unique_id][0] 
            self.borrowing = 0.      
            self.updateBlanceSheet()

        
    def step(self):
        if self.default == 0:
            self.borrowRequest()
    

class bankingSystem(mesa.Model):
    def __init__(self, params, seed = None):
        banksFile = params["banksFile"]
        leverageRatio = params["leverageRatio"]
        cashReserve = params["cashReserve"]
        num_borrowing = params["num_borrowing"]
        sizeOfBorrowing = params["sizeOfBorrowing"]
        num_banks = params["num_banks"]
        alpha = params["alpha"]
        beta = params["beta"]
        concentrationParameter = params["concentrationParameter"]
        fedRate = params["fedRate"]
        portfolioReturnRate = params["portfolioReturnRate"]
        returnVolatiliy = params["returnVolatiliy"]
        returnCorrelation = params["returnCorrelation"]
        shockSize = params["shockSize"]
        shockDuration = params["shockDuration"]


        # interest rate  
        self.fedRate = (fedRate+1)**(1/252) - 1
        # portfolio return rate
        self.portfolioReturnRate = (portfolioReturnRate+1)**(1/252) - 1
        # portfolio return volatility
        self.returnVolatiliy = returnVolatiliy/np.sqrt(252)
        # return correlation matrix
        cMatrix = np.ones((num_banks,num_banks))*returnCorrelation
        np.fill_diagonal(cMatrix, 1)
        self.Cholesky = np.linalg.cholesky(cMatrix * self.returnVolatiliy**2)
        # number of liquidity shocks
        # size of the shock
        self.shockSize = shockSize
        # time of the shock
        self.shockDuration = shockDuration
        # asset recovery rate 
        self.alpha = alpha
        # interbank loan recovery rate
        self.beta = beta
        
        # read in banks equity capital
        banksData = pd.read_csv(banksFile)
        self.banks = banksData["Name"]
        self.N = num_banks
        assert(self.N == len(self.banks))
        self.leverageRatio = leverageRatio
        self.cashReserve = cashReserve
        self.num_borrowing = num_borrowing
        self.sizeOfBorrowing = sizeOfBorrowing
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # we also introduce a time decay factor for trust       
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.N,self.N))
            np.fill_diagonal(self.concentrationParameter, 0.)
            self.trustMatrix = self.concentrationParameter / (self.N - 1)
        else:
            self.concentrationParameter = concentrationParameter
            self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
        # liability matrix 
        self.L = np.zeros((self.N,self.N))
        # asset matrix
        self.e = (banksData["Assets"].values).reshape(self.N,1)
        self.e_cash = (banksData["Cash"].values).reshape(self.N,1)
        self.e_sec = (banksData["Securities"].values).reshape(self.N,1)
        self.HTMper = (banksData["HTM"].values).reshape(self.N,1) / self.e_sec
        self.e_ill = self.e - self.e_cash - self.e_sec
        # deposit matrix
        self.d = banksData["Deposits"].values.reshape(self.N,1)
        # create a schedule for banks
        self.schedule = mesa.time.RandomActivation(self)
        
        # correlated shocks
        # set the bank's equity to drop
        numOfShocks = shockDuration[1] - shockDuration[0] + 1
        R = np.abs(shockSize * self.Cholesky @ np.random.randn(self.N,1))
        self.r = np.power(1 + R, 1.0/numOfShocks) - 1 
        
        # create banks and put them in schedule
        for i in range(self.N):
            a = Bank(i, self)
            a.cash = self.e_cash[i][0]
            a.sec = self.e_sec[i][0]
            a.ill = self.e_ill[i][0]
            a.deposit = self.d[i][0]
            a.updateBlanceSheet()
            self.schedule.add(a)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trust Matrix": "trustMatrix",
                             "Liability Matrix": "L",
                             "Asset Matrix": "e"},
            agent_reporters={ "asset": "asset",
                             "cash": "cash",
                             "sec": "sec",
                             "ill": "ill",
                             "Lending": "lending",
                             "Deposit": "deposit",
                             "Borrowing": "borrowing", 
                             "Equity": "equity",
                             "Default": "default",
                             "Leverage": "leverage"})
        
    def updateTrustMatrix(self):
        # add time decay of concentration parameter
        self.concentrationParameter = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True) * (self.N - 1) * self.num_borrowing
        self.trustMatrix = self.concentrationParameter / (self.N - 1) / self.num_borrowing
        
    def shockWaterFlow(self,shockSize):  
        # waterfall logic shock
        # cash shock 
        expectedCashReserve =  (self.e * self.cashReserve)
        maxCashShock = np.maximum(self.e_cash - self.e*expectedCashReserve, 0)
        actCashShock = np.minimum(maxCashShock, shockSize)
        self.e_cash -= actCashShock
        shockSize -= actCashShock 
        # securities shock
        maxSecShock = shockSize * self.HTMper * (1/0.8)
        actSecShock = np.minimum(self.e_sec, maxSecShock)
        self.e_sec -= actSecShock
        shockSize -= actSecShock / self.HTMper / (1/0.8)
        # illiquid shock    
        actIllShock = shockSize * (1/0.5)
        self.e_ill -= actIllShock
        shockSize = 0 
        self.e = self.e_cash + self.e_sec + self.e_ill
        
    def clearingDebt(self): 
        # Returns the new portfolio value after clearing debt
        _, e = eisenbergNoe(self.L*(1+self.fedRate), self.e, self.alpha, self.beta)
        assert((self.e>=0).all())
        assert((e>=0).all())
        drop = np.maximum(self.e - e, 0)
        self.shockWaterFlow(drop)
        insolventBanks = np.where(self.e - self.d <= 0)[0]
        # reset the Liabilities matrix after clearing debt
        self.L = np.zeros((self.N,self.N))
        if len(insolventBanks) > 0:
            self.concentrationParameter[:,insolventBanks] = 0
            self.e[insolventBanks] = 0
        for agent in self.schedule.agents:
            agent.reset()

    def returnOnPortfolio(self):
        # Return on the portfolio:
        R = self.portfolioReturnRate + (self.Cholesky @ np.random.randn(self.N,1)) 
        self.e += self.e * R
        self.e_cash += self.e_cash * R
        self.e_sec += self.e_sec * R
        self.e_ill += self.e_ill * R
    
    def depositShock(self):
        # liquidity shock to banks portfolio
        if self.schedule.time >= self.shockDuration[0] and self.schedule.time <= self.shockDuration[1]:
            # set the bank's portfolio to drop
            shockSize = self.r * self.d
            self.d -= shockSize
            self.shockWaterFlow(shockSize)
    
    def simulate(self):
        self.schedule.step()
        self.datacollector.collect(self)
        self.returnOnPortfolio()
        #self.depositShock()
        #self.clearingDebt()
        self.updateTrustMatrix()
