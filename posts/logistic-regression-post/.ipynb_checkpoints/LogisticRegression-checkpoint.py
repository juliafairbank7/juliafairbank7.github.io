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
            
            #update
            self.w_hat = (
                self.w_hat 
                - alpha 
                * self.gradient(X, y)
            )
            
            self.score_history.append(self.score(X, y))
            self.loss_history.append(self.empirical_risk(X, y))
        
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

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                grad = self.gradient(x_batch, y_batch) 
                
                #update
                self.w_hat = (
                self.w_hat 
                - alpha 
                * grad
                )
                
            self.score_history.append(self.score(X, y))
            self.loss_history.append(self.empirical_risk(X, y))
            
                                  
    def predict(self, X):
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)            
        return X_hat@self.w_hat

    def gradient(self, X, y):
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        sum_i = 0;
        n = X_hat.shape[0]
        
        for i in range(n):
            sum_i += (self.sigmoid(
                np.dot(self.w_hat, X_hat[i])
            )
                - y[i]
            ) * X_hat[i]
        
        grad = 1/n * sum_i
        
        return grad
                                  
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
                                  
    
    def score(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        predictions = self.predict(X)
        
        accuracy = predictions == y 
        
        accuracy = accuracy * 1
        
        accuracy = accuracy.mean()
        
        return accuracy
    
                                  
    def logistic_loss(self, y_hat, y): 
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def empirical_risk(self, X, y,):
        y_hat = self.predict(X)
        return self.logistic_loss(y_hat, y).mean()
