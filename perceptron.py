import numpy as np

class Perceptron:

    def fit(self, X, y, max_steps=1000):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
       
        #make -1 or +1 
        y_hat = y*2-1
        
        #initialize random w vector
        self.w_hat = np.random.rand(X_hat.shape[1]) 
        
        self.history = [] 
        
        for _ in range(max_steps):
            
            i = np.random.randint(0, X.shape[0])
            
            #update
            self.w_hat = (
                self.w_hat 
                + ((y_hat[i]*np.dot(self.w_hat, X_hat[i]) <0)*1)
                * y_hat[i]
                * X_hat[i]
            )
            
            loss = 1 - self.score(X, y)
            self.history.append(loss)
            
            if loss == 0:
                break
            

    def predict(self, X):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        prediction_vector = X_hat @ self.w_hat
        
        prediction_vector = prediction_vector > 0 #true false
        
        prediction_vector = prediction_vector * 1 #1 0
        
        return prediction_vector
            
    
    def score(self, X, y):
        #preprocess X by padding with 1s
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        predictions = self.predict(X)
        
        accuracy = predictions == y 
        
        accuracy = accuracy * 1
        
        accuracy = accuracy.mean()
        
        return accuracy
    