import numpy as np

class LogisticRegression:
    
    def fit(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #make -1 or +1 
        y_hat = y*2-1
    
    def predict(self, X):
        pass
    
    def score(self, X, y):
        pass
    
    def loss(self, X, y):
        pass