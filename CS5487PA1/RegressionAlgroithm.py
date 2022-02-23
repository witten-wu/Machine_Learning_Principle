import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def Load_data():
    sampx = np.loadtxt("./PA-1-data-text/polydata_data_sampx.txt")
    sampy = np.loadtxt("./PA-1-data-text/polydata_data_sampy.txt")
    polyx = np.loadtxt("./PA-1-data-text/polydata_data_polyx.txt")
    polyy = np.loadtxt("./PA-1-data-text/polydata_data_polyy.txt")
    SampX = sampx.reshape(len(sampx),1)
    SampY = sampy.reshape(len(sampy), 1)
    PolyX = polyx.reshape(len(polyx), 1)
    PolyY = polyy.reshape(len(polyy), 1)
    return SampX,SampY,PolyX,PolyY

def Load_data2():
    TestX = np.loadtxt("./PA-1-data-text/count_data_testx.txt")
    testy = np.loadtxt("./PA-1-data-text/count_data_testy.txt")
    TrainX = np.loadtxt("./PA-1-data-text/count_data_trainx.txt")
    trainy = np.loadtxt("./PA-1-data-text/count_data_trainy.txt")
    TestY = testy.reshape(len(testy), 1)
    TrainY = trainy.reshape(len(trainy), 1)
    return TrainX,TrainY,TestX,TestY

def vectorX(SampX,k):
    return np.array([SampX**i for i in range(k+1)]).reshape(k+1,1)
def vectorXF(SampX,k):
    Temp = np.array([vectorX(j,k) for j in SampX]).transpose()
    return np.matrix(Temp)

# 1 least-squares (LS)
def LSTheta(SampX,SampY):
    return np.dot(np.matrix(np.dot(SampX,SampX.transpose())).I,SampX).dot(SampY)

# 2 regularized LS (RLS)
def RLSTheta(SampX,SampY):
    lamda=1
    return (np.matrix(np.dot(SampX, SampX.transpose()) + lamda * np.identity(len(SampX))).I).dot(SampX).dot(SampY)

# Prediction
def Prediction(PolyX,theta):
    return np.dot(PolyX.transpose(),theta)

# 3 L1-regularized LS (LASSO)
# choose to use the third party
def LassoRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scatter", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))
    ])

# 4 robust regression (RR)
# choose to use the third party
def RRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scatter", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha))
    ])

# 5 Bayesian regression (BR)
def BRPosterior(SampX,SampY):
    alpha=1
    St = np.matrix((1 / alpha) * np.identity(len(SampX))+ (1 / 5) * np.dot(SampX,SampX.transpose())).I
    Mean = (1 / 5) * St.dot(SampX).dot(SampY)
    return St, Mean

def BRPrediction(PolyX, St, Mean):
    Meanx = np.dot(PolyX.transpose(), Mean)
    Covx = np.dot(PolyX.transpose(),St).dot(PolyX)
    return Meanx, Covx


# Error MSE
def MSEerror(PredY,PolyY):
    MSE = mean_squared_error(PolyY, PredY)
    return MSE

# Error MAE
def MAEerror(PredY,PolyY):
    MAE = mean_absolute_error(PolyY, PredY)
    return MAE

def CreatNewmatrix(value):
    for i in range(np.array(value).shape[0]):
        for j in range(np.array(value).shape[1]):
            value[i][j]=value[i][j]**2
    return value