import RegressionAlgroithm as RA
import matplotlib.pyplot as plt
import numpy as np
import random


if __name__=='__main__':
    k = 5
    [SampX,SampY,PolyX,PolyY]=RA.Load_data()
    Subset_Size = round(SampY.size * 0.1)
    Sel = random.sample(range(1, SampY.size), Subset_Size)
    for n in Sel:
        noise = np.random.randint(10,100)
        SampY[n] += noise
    
    vectorSampXF=RA.vectorXF(SampX, k)
    vectorPolyXF=RA.vectorXF(PolyX, k)
    
    # 1 least-squares (LS)
    LS_Theta=RA.LSTheta(vectorSampXF,SampY)
    LS_PredY=RA.Prediction(vectorPolyXF,LS_Theta)
    LS_MSEerror=RA.MSEerror(LS_PredY,PolyY)
    print('LS_MSE:',LS_MSEerror)
    plt.figure(1)
    plt.title('LS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX,LS_PredY)
    plt.scatter(SampX, SampY, color = "green", s = 10)

    # 2 regularized LS (RLS)
    RLS_Theta=RA.RLSTheta(vectorSampXF,SampY)
    RLS_PredY = RA.Prediction(vectorPolyXF, RLS_Theta)
    RLS_MSEerror = RA.MSEerror(RLS_PredY, PolyY)
    print('RLS_MSE:', RLS_MSEerror)
    plt.figure(2)
    plt.title('RLS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX,RLS_PredY)
    plt.scatter(SampX, SampY, color = "green", s = 10)

    # 3 L1-regularized LS (LASSO)
    lasso=RA.LassoRegression(degree=5, alpha=0.01)
    lasso.fit(SampX,SampY)
    lasso_pred = lasso.predict(PolyX)
    lasso_MSEerror = RA.MSEerror(lasso_pred, PolyY)
    print('LASSO_MSE:', lasso_MSEerror)
    plt.figure(3)
    plt.title('LASSO')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, lasso_pred)
    plt.scatter(SampX, SampY, color = "green", s = 10)

    # 4 robust regression (RR)
    ridge=RA.RRegression(degree=5, alpha=0.01)
    ridge.fit(SampX,SampY)
    ridge_pred = ridge.predict(PolyX)
    ridge_MSEerror = RA.MSEerror(ridge_pred, PolyY)
    print('RR_MSE:', ridge_MSEerror)
    plt.figure(4)
    plt.title('RR')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, ridge_pred)
    plt.scatter(SampX, SampY, color = "green", s = 10)

    # 5 Bayesian regression (BR)
    St, Mean=RA.BRPosterior(vectorSampXF, SampY)
    Meanx, Covx=RA.BRPrediction(vectorPolyXF, St, Mean)
    BR_MSEerror = RA.MSEerror(Meanx, PolyY)
    print('BR_MSE:',BR_MSEerror)
    Std_err = np.sqrt(Covx)
    plt.figure(5)
    plt.title('BR')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, Meanx, 'r')
    plt.plot(PolyX, Meanx-Std_err)
    plt.plot(PolyX, Meanx+Std_err)
    plt.plot(SampX, SampY, 'b.')
    #plt.plot(PolyX, Meanx)
    #plt.plot(PolyX, Std_err)
    #plt.scatter(SampX, SampY, color = "green", s = 10)
    plt.show()
