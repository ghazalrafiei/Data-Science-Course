import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100, momentum = 0.9, limit = 0.00001, stop_step = 5):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.momentum = momentum
        self.JHist = []
        self.limit = limit
        self.stop_step = stop_step

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,_ = X.shape
        cost_0 = 0
        cost_1 = 1
        diff = cost_1-cost_0
        i = 0
        increasing = 0
        while abs(diff)>self.limit and increasing<self.stop_step and i<self.n_iter:
            c = self.computeCost(X, y, theta)
            i+=1
            cost_0 = cost_1
            cost_1 = c
            diff = cost_1-cost_0
            if diff>0:
                increasing +=1
            else:
                increasing = 0
            
            h = np.dot(X,theta)
            loss = np.subtract(h,y)
            gradient = np.dot(X.T, loss)/n
            theta -= self.alpha*gradient

            self.alpha*=self.momentum
            self.JHist.append((c, theta))
            print(f"Iteration: {i+1:3}, Cost: {c:4f}, alpha: {self.alpha:4f}")

        print('Stop reason:')
        if increasing >= self.stop_step:
            print('Cost started to increase.')
        if abs(diff) < self.limit :
            print('Cost decreasing was vanishing.')
        if i>=self.n_iter:
            print('Iterations finished.')
            
        return theta

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
               make certain you don't return a matrix with just one value! 
        '''
        now = np.dot(X,theta)
        cost = np.sqrt(np.mean(np.square(np.subtract(now,y))))
        return cost

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        _,d = X.shape
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        return np.dot(X, self.theta)
    
    def plot_cost(self):
        costs = [self.JHist[i][0] for i in range(len(self.JHist))]
        sns.lineplot(y=costs, x = range(len(costs)))
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.show()