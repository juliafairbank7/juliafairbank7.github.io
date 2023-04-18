import numpy as np
from scipy.optimize import minimize
np.seterr(all='ignore')

class LogisticRegression:
    
    def fit(self, X, y, alpha=0.1, max_epochs=1000):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1]) 
        
        # list of the evolution of the score over the training period
        self.score_history = []
        self.loss_history = []
        
        n = X.shape[0]
        
        # compute complete gradient
        for _ in range(max_epochs):
            
            i = np.random.randint(0, n)
            
            gradient = self.gradient(X_hat,y)
            self.w_hat -= alpha * gradient
            
            curr_loss = self.logistic_loss(X_hat,y)
           
            self.loss_history = np.append(self.loss_history, curr_loss)
            self.score_history = np.append(self.score_history, self.score(X_hat,y))
            
            if (np.isclose(gradient.all(),0)):
                done = True
            
        
    def fit_stochastic(self, X, y, m_epochs=1000, momentum = False, batch_size = 10, alpha = .1):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1]) 
        
        # list of the evolution of the score over the training period
        self.score_history = []
        self.loss_history = []
        
        n = X.shape[0]
            
        for j in np.arange(m_epochs):
            order = np.arange(n)
            np.random.shuffle(order)
            prev_loss = np.inf

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_hat[batch,:]
                y_batch = y[batch]
                
                gradient = self.gradient(x_batch,y_batch)
                self.w_hat -= alpha * gradient
                
                curr_loss = self.logistic_loss(X_hat,y)
            
                if (np.isclose(curr_loss,0) or np.isnan(curr_loss)):
                    break
                    
                if curr_loss < prev_loss:
                    prev_loss = curr_loss
                    
            self.loss_history = np.append(self.loss_history, curr_loss)
            self.score_history = np.append(self.score_history, self.score(X_hat,y))
            
                                  
    def predict(self, X):
        return 1*(X@self.w_hat)>0

    def gradient(self, X, y):
        sigmoid = self.sigmoid(X@self.w_hat)
        return np.mean(((sigmoid - y)[:,np.newaxis]*X),axis=0)
                       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
                                  
    def score(self, X, y):
        return (y== self.predict(X)).mean()
                                  
    def logistic_loss(self, X, y):
        y_hat = X@self.w_hat
        return (-y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))).mean()
