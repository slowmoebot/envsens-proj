import numpy as np


class RLS:
    def __init__(self):
        """
        RLS initialisation
        """
        P_0 = np.load("temp_P0.npy") 
        w_0 = np.load("temp_w0.npy")
        
        self.P = P_0 * np.eye(2)  # state matrix, 2x2
        self.w = w_0 * np.ones((2, 1))  # weights (coefficients of the linear model including constant, 2x1)
        self.error = 0
        self.dx = 0
        self.use_prediction_error = np.load("temp_err.npy") == 1
        

    def update(self, x: float, y: float):
        """
        RLS state update
        :param x: past/input observation (scalar)
        :param y: current/output observation (scalar)
        """

        X = np.reshape([1, x], (1, 2))  # reshape to a 1x2 matrix
        alpha = float(y - X @ self.w)
        g = (self.P @ X.T) / (1 + X @ self.P @ X.T)
        if np.any(np.isnan(g * alpha)):
            print(x,y)
        self.w = self.w + g * alpha
        self.P = self.P - g * X * self.P
        self.dx = np.sqrt((self.dx*self.w[1])**2+(x*self.P[1][1])**2+self.P[0][0]**2)[0]
        if self.use_prediction_error:
            self.error = self.dx
        else:
            self.error = np.abs(alpha)

    def predict(self, x: float) -> float:
        """
        Predict observation, using RLS model
        :param x: past observation (scalar)
        :return: predicted observation (scalar)
        """

        X = np.reshape([1, x], (1, 2))  # reshape to a 1x2 matrix
        return float(X @ self.w)
