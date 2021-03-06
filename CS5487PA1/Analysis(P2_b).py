import RegressionAlgroithm as RA
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    #[SampX,SampY,PolyX,PolyY]=RA.Load_data2()
    [A1,B1,C1,D1]=RA.Load_data2()
    Temp = np.array(A1, copy=True)
    SampX_square = np.square(A1)
    Temp = np.append(Temp, SampX_square, axis=0)
    SampX = Temp   
    SampY = B1

    Temp = np.array(C1, copy=True)
    TestX_square = np.square(C1)
    Temp = np.append(Temp, TestX_square, axis=0)
    PolyX = Temp
    PolyY = D1

    # 1 least-squares (LS)
    LS_Theta=RA.LSTheta(SampX,SampY)
    LS_PredY=RA.Prediction(PolyX,LS_Theta)
    LS_MSEerror=RA.MSEerror(LS_PredY,PolyY)
    print('LS_MSE:',LS_MSEerror)
    LS_MAEerror=RA.MAEerror(LS_PredY,PolyY)
    print('LS_MAE:',LS_MAEerror)
    plt.figure(1)
    plt.title('LS')
    plt.plot(LS_PredY, 'g', label='predict')
    plt.plot(PolyY, 'b', label='true')
    plt.legend()

    # 2 regularized LS (RLS)
    RLS_Theta=RA.RLSTheta(SampX,SampY)
    RLS_PredY = RA.Prediction(PolyX, RLS_Theta)
    RLS_MSEerror = RA.MSEerror(RLS_PredY, PolyY)
    print('RLS_MSE:', RLS_MSEerror)
    RLS_MAEerror=RA.MAEerror(RLS_PredY,PolyY)
    print('RLS_MAE:',RLS_MAEerror)
    plt.figure(2)
    plt.title('RLS')
    plt.plot(RLS_PredY, 'g', label='predict')
    plt.plot(PolyY, 'b', label='true')
    plt.legend()

    # 3 L1-regularized LS (LASSO)
    lasso=RA.LassoRegression(degree=1,alpha=0.01)
    lasso.fit(SampX.transpose(),SampY)
    lasso_pred = lasso.predict(PolyX.transpose())
    lasso_MSEerror = RA.MSEerror(lasso_pred, PolyY)
    print('LASSO_MSE:', lasso_MSEerror)
    lasso_MAEerror = RA.MAEerror(lasso_pred, PolyY)
    print('LASSO_MAE:', lasso_MAEerror)
    plt.figure(3)
    plt.title('LASSO')
    plt.plot(lasso_pred, 'g', label='predict')
    plt.plot(PolyY, 'b', label='true')
    plt.legend()

    # 4 robust regression (RR)
    ridge=RA.RRegression(degree=1,alpha=0.01)
    ridge.fit(SampX.transpose(),SampY)
    ridge_pred = ridge.predict(PolyX.transpose())
    ridge_MSEerror = RA.MSEerror(ridge_pred, PolyY)
    print('RR_MSE:', ridge_MSEerror)
    ridge_MAEerror = RA.MAEerror(ridge_pred, PolyY)
    print('RR_MAE:', ridge_MAEerror)
    plt.figure(4)
    plt.title('RR')
    plt.plot(ridge_pred, 'g', label='predict')
    plt.plot(PolyY, 'b', label='true')
    plt.legend()

    # 5 Bayesian regression (BR)
    St, Mean=RA.BRPosterior(SampX, SampY)
    Meanx, Covx=RA.BRPrediction(PolyX, St, Mean)
    BR_MSEerror = RA.MSEerror(Meanx, PolyY)
    print('BR_MSE:',BR_MSEerror)
    BR_MAEerror = RA.MAEerror(Meanx, PolyY)
    print('BR_MAE:',BR_MAEerror)
    Std_err = np.sqrt(Covx)
    plt.figure(5)
    plt.title('BR')
    plt.plot(Meanx, 'g', label='predict')
    plt.plot(PolyY, 'b', label='true')
    plt.legend()
    plt.show()
