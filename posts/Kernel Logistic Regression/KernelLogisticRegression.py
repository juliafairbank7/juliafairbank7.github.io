import numpy as np
from scipy.optimize import minimize
np.seterr(all='ignore')

class KernelLogisticRegression:
    
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
    
    def fit(self, X, y, alpha=0.1, max_epochs=1000):
        #pad X, save as instance to X_train
        self.X_train = np.append(X, np.ones((X.shape[0], 1)), 1)
        km = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        w0 = np.random.rand(self.X_train.shape[0]) - 0.5 #init v
        result = minimize(lambda w: self.empirical_risk(X, y, w), x0 = w0) 
        self.v = result.x 
                                     
    def predict(self, X):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        km = self.kernel(self.X_train, X_, **self.kernel_kwargs)
        ip = np.dot(self.v, km)
        ip_binary = (ip>0) *1
        return ip_binary
            
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

    def empirical_risk(self, X, y, v):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        km = self.kernel(self.X_train, X_, **self.kernel_kwargs)
        ip = np.dot(v, km)
        return self.logistic_loss(ip, y).mean()
