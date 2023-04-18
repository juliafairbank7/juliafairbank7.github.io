import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
np.seterr(all='ignore')

class LinearRegression:
    
    def fit_analytic(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1]) 
        
        self.w_hat = inv(X_hat.T@X_hat)@X_hat.T@y
    
    def w(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1]) 
        self.w_hat = inv(X_hat.T@X_hat)@X_hat.T@y
        
        return self.w_hat
        
    def fit_gradient(self, X, y, alpha=0.001, max_epochs=1000):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1], 1) 
        
        self.score_history = []
        
        # compute complete gradient
        for _ in range(int(max_epochs)):
            grad = self.gradient(X, y)
            #update
            self.w_hat = (
                self.w_hat 
                - alpha 
                * grad
            )
            self.score_history.append(self.score(X, y))

    def predict(self, X):
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)       
        
        return X_hat@self.w_hat

    def gradient(self, X, y):
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        y = y.reshape(-1, 1)
        
        P = X_hat.T@X_hat
        q = X_hat.T@y
        grad = 2 * (P@self.w_hat - q)
        
        return grad
    
    def score(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        y = y.reshape(-1, 1)
        y_bar = y.mean()
        predictions = self.predict(X)
        num = ((predictions - y)**2).sum()
        denom = ((y_bar - y)**2).sum()
        c = 1 - (num/denom)
        
        return c
