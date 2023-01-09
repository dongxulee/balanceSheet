from bankingSystem import *

def action(information, w):
    '''
        Gaussian policy function based on the information of the lender and the borrower
        # all the information needed to make the decision
        # asset
        lender.lending/lender.portfolio       
        # liabilities
        lender.borrowing/lender.portfolio

        # asset      
        borrower.lending/borrower.portfolio            
        # liabilities
        borrower.borrowing/borrower.portfolio      
    '''
    return np.dot(w, information) + np.random.normal(0, 0.01)