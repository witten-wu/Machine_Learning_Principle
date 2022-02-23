import RegressionAlgroithm as RA
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__=='__main__':
    k = 5
    [SampX,SampY,PolyX,PolyY]=RA.Load_data()
    Subset_Size = round(SampX.size * 0.5)
    Sel = random.sample(range(1, SampX.size), Subset_Size)
    SampX_Subset = []
    SampY_Subset = []
    for n in Sel:
        SampX_Subset.append(SampX[n])
        SampY_Subset.append(SampY[n])
    SubSampX = np.array(SampX_Subset).reshape(len(SampX_Subset), 1)
    SubSampY = np.array(SampY_Subset).reshape(len(SampY_Subset), 1)

    vectorSampXF=RA.vectorXF(SubSampX, k)
    vectorPolyXF=RA.vectorXF(PolyX, k)
    
    # 1 least-squares (LS)
    LS_Theta=RA.LSTheta(vectorSampXF,SubSampY)
    LS_PredY=RA.Prediction(vectorPolyXF,LS_Theta)
    LS_MSEerror=RA.MSEerror(LS_PredY,PolyY)
    print('LS_MSE:',LS_MSEerror)
    plt.figure(1)
    plt.title('LS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX,LS_PredY)
    plt.scatter(SubSampX, SubSampY, color = "green", s = 10)

    # 2 regularized LS (RLS)
    RLS_Theta=RA.RLSTheta(vectorSampXF,SubSampY)
    RLS_PredY = RA.Prediction(vectorPolyXF, RLS_Theta)
    RLS_MSEerror = RA.MSEerror(RLS_PredY, PolyY)
    print('RLS_MSE:', RLS_MSEerror)
    plt.figure(2)
    plt.title('RLS')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX,RLS_PredY)
    plt.scatter(SubSampX, SubSampY, color = "green", s = 10)

    # 3 L1-regularized LS (LASSO)
    lasso=RA.LassoRegression(degree=5, alpha=0.01)
    lasso.fit(SubSampX,SubSampY)
    lasso_pred = lasso.predict(PolyX)
    lasso_MSEerror = RA.MSEerror(lasso_pred, PolyY)
    print('LASSO_MSE:', lasso_MSEerror)
    plt.figure(3)
    plt.title('LASSO')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, lasso_pred)
    plt.scatter(SubSampX, SubSampY, color = "green", s = 10)

    # 4 robust regression (RR)
    ridge=RA.RRegression(degree=5, alpha=0.01)
    ridge.fit(SubSampX,SubSampY)
    ridge_pred = ridge.predict(PolyX)
    ridge_MSEerror = RA.MSEerror(ridge_pred, PolyY)
    print('RR_MSE:', ridge_MSEerror)
    plt.figure(4)
    plt.title('RR')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, ridge_pred)
    plt.scatter(SubSampX, SubSampY, color = "green", s = 10)

    # 5 Bayesian regression (BR)
    St, Mean=RA.BRPosterior(vectorSampXF, SubSampY)
    Meanx, Covx=RA.BRPrediction(vectorPolyXF, St, Mean)
    Std_err = np.sqrt(Covx)
    BR_MSEerror = RA.MSEerror(Meanx, PolyY)
    print('BR_MSE:',BR_MSEerror)
    plt.figure(5)
    plt.title('BR')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(PolyX, Meanx, 'r')
    plt.plot(PolyX, Meanx-Std_err)
    plt.plot(PolyX, Meanx+Std_err)
    plt.plot(SubSampX, SubSampY, 'b.')
    # plt.plot(PolyX, Meanx)
    # plt.plot(PolyX, Std_err)
    # plt.scatter(SubSampX, SubSampY, color = "green", s = 10)
    plt.show()

    # Calculate the error versus training size
    MSE_LS = []
    MSE_RLS = []
    MSE_LASSO = []
    MSE_RR = []
    MSE_BR = []
    Subset_Ratio = [0.1,0.25,0.5,0.75]
    for i in Subset_Ratio:
        Subset_Size = round(SampX.size * i)
        A = []
        B = []
        C = []
        D = []
        E = []
        for j in [0,1,2,3,4,5,6,7,8,9]:
            Sel = random.sample(range(1, SampX.size), Subset_Size)
            SampX_Subset = []
            SampY_Subset = []
            for n in Sel:
                SampX_Subset.append(SampX[n])
                SampY_Subset.append(SampY[n])
            SubSampX = np.array(SampX_Subset).reshape(len(SampX_Subset), 1)
            SubSampY = np.array(SampY_Subset).reshape(len(SampY_Subset), 1)

            vectorSampXF=RA.vectorXF(SubSampX, k)
            vectorPolyXF=RA.vectorXF(PolyX, k)

            # 1 least-squares (LS)
            LS_Theta=RA.LSTheta(vectorSampXF,SubSampY)
            LS_PredY=RA.Prediction(vectorPolyXF,LS_Theta)
            LS_MSEerror=RA.MSEerror(LS_PredY,PolyY)
            A.append(LS_MSEerror)

            # 2 regularized LS (RLS)
            RLS_Theta=RA.RLSTheta(vectorSampXF,SubSampY)
            RLS_PredY = RA.Prediction(vectorPolyXF, RLS_Theta)
            RLS_MSEerror = RA.MSEerror(RLS_PredY, PolyY)
            B.append(RLS_MSEerror)

            # 3 L1-regularized LS (LASSO)
            lasso=RA.LassoRegression(degree=5, alpha=0.01)
            lasso.fit(SubSampX,SubSampY)
            lasso_pred = lasso.predict(PolyX)
            lasso_MSEerror = RA.MSEerror(lasso_pred, PolyY)
            C.append(lasso_MSEerror)

            # 4 robust regression (RR)
            ridge=RA.RRegression(degree=5, alpha=0.01)
            ridge.fit(SubSampX,SubSampY)
            ridge_pred = ridge.predict(PolyX)
            ridge_MSEerror = RA.MSEerror(ridge_pred, PolyY)
            D.append(ridge_MSEerror)

            # 5 Bayesian regression (BR)
            St, Mean=RA.BRPosterior(vectorSampXF, SubSampY)
            Meanx, Covx=RA.BRPrediction(vectorPolyXF, St, Mean)
            Std_err = np.sqrt(Covx)
            BR_MSEerror = RA.MSEerror(Meanx, PolyY)
            E.append(BR_MSEerror)
        
        # calculate the average MSE
        MSE_LS.append(sum(A)/len(A))
        MSE_RLS.append(sum(B)/len(B))
        MSE_LASSO.append(sum(C)/len(C))
        MSE_RR.append(sum(D)/len(D))
        MSE_BR.append(sum(E)/len(E))
    print(MSE_LS)
    print(MSE_RLS)
    print(MSE_LASSO)
    print(MSE_RR)
    print(MSE_BR)

    plt.plot(Subset_Ratio,MSE_LS,label='LS',color='g')
    plt.plot(Subset_Ratio,MSE_RLS,label='RLS',color='b')
    plt.plot(Subset_Ratio,MSE_LASSO,label='LASSO',color='y')
    plt.plot(Subset_Ratio,MSE_RR,label='RR',color='r')
    plt.plot(Subset_Ratio,MSE_BR,label='BR',color='m')
    plt.legend()
    plt.ylim([0,200])
    plt.xlabel('Size')
    plt.ylabel('MSE')
    plt.show()



