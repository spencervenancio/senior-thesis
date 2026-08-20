import numpy as np
import pandas as pd

"""
Datasets: [XORModel, LinearAdditiveModel, NonlinearAdditiveModel, ConditionalInteractionModel,
            LogisticModel]
"""

def XORModel(n=1_000, noise=0.1):
    ep1 = np.random.normal(0, noise, n)
    ep2 = np.random.normal(0, noise, n)
    x1 = np.random.uniform(low=-1, high=1, size=n)
    x2 = np.random.uniform(low=-1, high=1, size=n)
    y = (x1 * x2) < 0
    return pd.DataFrame({'x1': x1 + ep1, 'x2': x2 + ep2, 'y': y})

def LinearAdditiveModel(n=1_000, noise=0.1):
    ep = np.random.normal(0, noise, n)
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)
    x7 = np.random.randn(n)
    x8 = np.random.randn(n)
    x9 = np.random.randn(n)
    x10 = np.random.randn(n)

    y = 2*x1 + x3 + 9*x5 + 9*x6 + 3*x7 + x10 + ep
    
    
    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                         'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 
                         'y': y})

def NonlinearAdditiveModel(n=1_000, noise=0.01):
    ep = np.random.normal(0, noise, n)
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)
    x7 = np.random.randn(n)
    x8 = np.random.randn(n)
    x9 = np.random.randn(n)
    x10 = np.random.randn(n)

    y = 2*(x1**2) + x3**3 + 9*np.sin(x5) + 9*np.exp(x6) + 3*np.cos(x7) + np.abs(x10) + ep
    

    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                         'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 
                         'y': y})

def ConditionalInteractionModel(n=1_000, noise=0.1):
    ep = np.random.normal(0, noise, n)
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)
    x7 = np.random.randn(n)
    x8 = np.random.randn(n)
    x9 = np.random.randn(n)
    x10 = np.random.randn(n)

    y = 2*(x1*x2)*(x3>0) + (x4*x5)*(x3<0) + 9*(x6*x7)*(x8>0) + (x9*x10)*(x8<0) + ep
    

    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                         'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 
                         'y': y})

def LogisticModel(n=1_000, noise=0.1):
    
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))

    ep = np.random.normal(0, noise, n)
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)
    x6 = np.random.randn(n)
    x7 = np.random.randn(n)
    x8 = np.random.randn(n)
    x9 = np.random.randn(n)
    x10 = np.random.randn(n)

    y = sigmoid(x1**2 + x3*x4 + 4*x6 + 7*(x8**2) + 2*(x9**3)) + ep
    

    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
                         'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 
                         'y': y})