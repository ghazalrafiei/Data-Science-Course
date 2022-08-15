import numpy as np

from linreg import LinearRegression
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    # load the data
    filePath = 'data/multivariateData.dat'
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')
    # print(allData)
    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 500
    alpha = 0.1
    momentum = 0.99
    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter, momentum=momentum)
    lr_model.fit(X,y)
    lr_model.plot_cost()    

    """Test Holdout"""
    test_path = r'data/holdout.npz'

    test = np.load(test_path)['arr_0']
    X_test = np.matrix(test[:, :-1])
    y_test = np.matrix(test[:, -1]).T

    X_test = (X_test-mean)/std

    # Add a row of ones for the bias term
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    y_test_predict = lr_model.predict(X_test)
    error = mean_squared_error(y_test, y_test_predict,squared=False)
    print(error)