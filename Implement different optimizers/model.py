from pickle import TRUE
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from tkinter.messagebox import NO
from sklearn.datasets import make_regression
import numpy as np


class Regressor:

    def __init__(self) -> None:
        self.n_samples = 200
        self.n_feature = 1
        self.X, self.y = self.generate_dataset(
            n_samples=self.n_samples, n_features=self.n_feature)
        _, d = self.X.shape
        self.theta = np.zeros((d, self.n_feature))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples,
                               n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.theta)
        return y

    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.theta)
        return y

    def compute_cost(self, theta):
        now = np.dot(self.X, theta)
        cost = np.sqrt(np.mean(np.square(np.subtract(now, self.y))))
        return cost

    def compute_loss(self):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression()
        cost = np.sqrt(np.mean(np.square(np.subtract(predictions, self.y))))
        return cost

    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)/self.X.shape[0]
        return grad

    def fit(self, optimizer='gd', n_iters=500,
            render_animation=False, alpha=0.001,
            momentum=0.9, batch_size=5, g=0, epsilon=0.1,
            m=0, v=0, beta1=0.9, beta2=0.8,verbose=True):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """

        figs = []
        loss = []
        cost_0 = 1e5
        cost_1 = 1e5
        increasing = 0

        slight_diff = 0
        k = 0
        self.b = 0.0
        self.m=0.0
        self.g = g
        for i in range(1, n_iters+1):
            k += 1

            if optimizer == 'gd':
                self.gradient_descent(alpha)

            elif optimizer == "sgd":
                cost_1=self.sgd_optimizer(batch_size=batch_size, alpha=alpha)

            elif optimizer == "sgdMomentum" or optimizer == 'sgdm':
                cost_1=self.sgd_momentum(batch_size=batch_size, alpha=alpha)
                alpha *= momentum

            elif optimizer == "adagrad":
                self.adagrad_optimizer(self.g, epsilon)

            elif optimizer == "rmsprop":
                self.rmsprop_optimizer(self.g, alpha, epsilon)

            elif optimizer == "adam":
                self.adam_optimizer(m, v, beta1, beta2, epsilon)

            cost_0 = cost_1

            if optimizer[0]!='s':
                cost_1 = self.compute_loss()

            if cost_1 > cost_0:
                increasing += 1
            else:
                increasing = 0

            if increasing == 10:
                print('cost started to increase')
                break

            if abs(cost_1-cost_0) < 0.001:
                slight_diff += 1
            else:
                slight_diff = 0

            if slight_diff == 10:
                print("cost decreasing started to vanish")
                break

            loss.append(cost_1)
            if i % 1 == 0 and verbose:
                print("Iteration: ", i)
                print("Loss: ", cost_1)

                if render_animation:

                    fig = plt.figure()
                    plt.scatter(self.X, self.y, color='red')
                    plt.plot(self.X, self.predict(self.X), color='blue')
                    plt.xlim(self.X.min(), self.X.max())
                    plt.ylim(self.y.min(), self.y.max())
                    plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                    plt.close()
                    figs.append(mplfig_to_npimage(fig))

        if render_animation and len(figs) > 0:
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'./figs/{optimizer}_animation.gif', fps=5)

        self.plot_path()
        self.plot_scatter(optimizer, k)

        return loss

    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        grad = self.compute_gradient()
        self.theta = self.theta - alpha*grad

    def sgd_optimizer(self, batch_size, alpha):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """

        indexes = np.random.randint(0, len(self.X), batch_size)
        
        X_new = self.X[indexes,:]
        y_new = self.y[indexes]
        N = len(X_new)
        
        f = y_new - (self.m*X_new + self.b)
        
        self.m -= alpha * (-2 * X_new.T.dot(f).sum() / N)
        self.b -= alpha * (-2 * f.sum() / N)

        return np.sqrt(np.mean(np.square(np.subtract(self.y, self.m*self.X+self.b))))
        

    def sgd_momentum(self, alpha, batch_size):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        indexes = np.random.randint(0, len(self.X), batch_size)
        
        X_new = self.X[indexes,:]
        y_new = self.y[indexes]
        N = len(X_new)
        
        f = y_new - (self.m*X_new + self.b)
        
        self.m -= alpha * (-2 * X_new.T.dot(f).sum() / N)
        self.b -= alpha * (-2 * f.sum() / N)

        return np.sqrt(np.mean(np.square(np.subtract(self.y, self.m*self.X+self.b))))
        

    def adagrad_optimizer(self, g, epsilon):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        f = 1e-5
        grad = self.compute_gradient()
        self.g += grad**2
        adjusted_grad = epsilon * grad / (f + np.sqrt(self.g))
        self.theta = self.theta-adjusted_grad

        return self.theta

    def rmsprop_optimizer(self, g, alpha, epsilon):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        f = 1e-5
        grad = self.compute_gradient()
        self.g = alpha*self.g + (1-alpha)*grad**2
        self.theta -= epsilon*grad/(np.sqrt(self.g)+f)

        return self.theta

    def adam_optimizer(self, m, v, beta1, beta2, epsilon):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        f = 1e-5
        grad = self.compute_gradient()
        m = beta1 * m + (1-beta1)*grad
        v = beta2 * v + (1-beta2)*grad**2
        self.theta -= epsilon*m / (np.sqrt(v)+f)

        return self.theta

    def plot_scatter(self, optimizer, iters):
        plt.scatter(self.X, self.y, color='red')
        plt.plot(self.X, self.predict(self.X), color='blue')
        plt.xlim(self.X.min(), self.X.max())
        plt.ylim(self.y.min(), self.y.max())
        plt.title(f'Optimizer:{optimizer}\nIteration: {iters}')
        fname = f'./figs/{optimizer}-final-regression.png'
        plt.savefig(fname)
        print(f'plot saved into {fname}')
        plt.close()

    def plot_path(self):
        """
        Plots the gradient descent path for the loss function
        Useful links: 
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """
        # TODO: Bonus!
