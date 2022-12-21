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
        # leverage ratio
        self.leverage = 0
        # if a bank is solvent
        self.default = False
    
    def updateBlanceSheet(self):
        self.equity = self.portfolio + self.lending - self.borrowing
        self.leverage = (self.portfolio + self.lending) / self.equity
        
    def borrowRequest(self):
        if not self.default:
            for _ in range(self.model.num_borrowing):
                if self.leverage < self.model.targetLeverageRatio:
                    # randomly choose a bank to borrow from the trust matrix
                    target = np.random.choice(self.model.N, p=self.model.trustMatrix[self.unique_id])
                    if target != self.unique_id:
                        # choose a borrowing amount equal to the equity capital
                        amount = self.equity
                        # bring out the target bank and let him decide whether to lend
                        other_agent = self.model.schedule.agents[target]
                        # if the lending decision is made, update the balance sheet
                        if other_agent.lendDecision(self, amount):
                            self.borrowingTarget.append(other_agent.unique_id)
                            self.borrowingAmount.append(amount)
                            self.borrowing = np.sum(self.borrowingAmount)
                            self.portfolio = self.portfolio + amount
                            self.updateBlanceSheet()
                            self.model.borrowingCollection.append([self.unique_id, target, amount])
                

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
            
    def payDebt(self):
        # if the bank is solvent
        if not self.default:
            # calcualate how much debt to pay % of the total debt
            numDebtToPay = int(len(self.borrowingTarget) * self.model.debtClearingRate)
            if numDebtToPay >= 1:
                debtToPayTarget = self.borrowingTarget[:numDebtToPay] 
                debtToPayAmount = self.borrowingAmount[:numDebtToPay]
                self.borrowingTarget = self.borrowingTarget[numDebtToPay:]
                self.borrowingAmount = self.borrowingAmount[numDebtToPay:]
                # changes in its own balance sheet 
                self.portfolio = self.portfolio - np.sum(debtToPayAmount)*(1+self.model.fedRate)
                self.borrowing = np.sum(self.borrowingAmount)
                self.updateBlanceSheet()
                # if the bank can not pay the debt or the equity capital is negative, the bank will default
                if self.portfolio < 0 or self.equity < 0:
                    self.default = True
                    self.model.concentrationParameter[:, self.unique_id] = 0
                    # all debts will be defaulted
                    for target, amount in zip(self.borrowingTarget + debtToPayTarget, self.borrowingAmount + debtToPayAmount):
                        # changes in couter-party's balance sheet
                        other_agent = self.model.schedule.agents[target]
                        other_agent.portfolio = other_agent.portfolio # not payment in this case amount * (1+self.model.fedRate)
                        debtIndex = list(zip(other_agent.lendingTarget, other_agent.lendingAmount)).index((self.unique_id, amount))
                        # cancel debt in counter-party's balance sheet
                        del other_agent.lendingTarget[debtIndex]
                        del other_agent.lendingAmount[debtIndex]
                        # update counter-party's leverage ratio
                        other_agent.lending = np.sum(other_agent.lendingAmount)
                        other_agent.updateBlanceSheet() 
                else:
                    for target, amount in zip(debtToPayTarget, debtToPayAmount):
                        # changes in couter-party's balance sheet
                        other_agent = self.model.schedule.agents[target]
                        other_agent.portfolio = other_agent.portfolio + amount * (1+self.model.fedRate)
                        debtIndex = list(zip(other_agent.lendingTarget, other_agent.lendingAmount)).index((self.unique_id, amount))
                        # cancel debt in counter-party's balance sheet
                        del other_agent.lendingTarget[debtIndex]
                        del other_agent.lendingAmount[debtIndex]
                        # update counter-party's leverage ratio
                        other_agent.lending = np.sum(other_agent.lendingAmount)
                        other_agent.updateBlanceSheet()
                    
    def returnOnPortfolio(self):
        if not self.default:
            self.portfolio = self.portfolio * (1+self.model.portfolioReturnRate)
            self.updateBlanceSheet()
    
    def step(self):
        if not self.default:
            self.returnOnPortfolio()
            self.borrowRequest()
            self.payDebt()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, targetLeverageRatio, num_borrowing, num_banks, 
                 debtClearingRate, concentrationParameter = None, fedRate = 0.00, portfolioReturnRate = 0.00):
        # enable batch run
        self.running = True
        # interest rate
        self.fedRate = fedRate
        # portfolio return rate
        self.portfolioReturnRate = portfolioReturnRate
        # debt clearing rate
        self.debtClearingRate = debtClearingRate
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
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.N,self.N))
        else:
            self.concentrationParameter = concentrationParameter
        np.fill_diagonal(self.concentrationParameter, 0)
        self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
        self.schedule = mesa.time.RandomActivation(self)
    
        # create banks and put them in schedule
        for i in range(self.N):
            a = Bank(i, self)
            self.schedule.add(a)
            a.portfolio = banksData["equity"][i]
            a.equity = banksData["equity"][i]
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trust Matrix": "trustMatrix"},
            agent_reporters={"Portfolio Value": "portfolio",
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
            self.concentrationParameter[borrowerIndex, lenderIndex] += 1
            self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
            # clean the borrowing collection
            self.borrowingCollection = []

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.updateTrustMatrix()
