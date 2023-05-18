import numpy as np

class LinearRegression():

    def predict(self, X): 
        
        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        return X_hat@self.w

    def fit_analytic(self, X, y):

        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        w = np.random.random((X_hat.shape[1]-1))
        self.w = w
        self.w = np.linalg.inv(X_hat.T@X_hat)@X_hat.T@y 

    def fit_gradient(self, X, y, alpha, max_epochs):

        X_hat = np.append(X, np.ones((X.shape[0], 1)), 1)
        w = np.random.random(X_hat.shape[1])
        self.w = w
        
        score_history = []
        self.score_history = score_history

        P = (X_hat.T).dot(X_hat)
        self.P = P

        q = (X_hat.T).dot(y)
        self.q = q

        loss = np.inf
        self.loss = loss

        for i in range(max_epochs):

            grad = self.gradient(self.P, self.q, self.w)
            self.grad = grad

            self.w = self.w - (alpha * self.grad)

            self.score_history.append(self.score(X, y))

            y_pred = self.predict(X)
                
            self.loss = np.mean((y-y_pred)**2)

    def score(self, X, y):
        
        y_hat = self.predict(X)
        return 1- ((np.sum((y_hat-y)**2))/(np.sum((y-y.mean())**2)))

    def gradient(self, P, q, w):

        gradient = (P@w - q)
        return gradient