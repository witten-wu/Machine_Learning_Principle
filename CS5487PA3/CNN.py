import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

def LoadFile(name):
    file = open(name,"r")
    Matrix = []
    for line in file:
        l = line.split()
        Tmplist = [float(v) for v in l]
        Matrix.append(Tmplist)
    Matrix = np.array(Matrix)
    file.close()
    return Matrix    
    
if __name__ == "__main__":
    DataX = LoadFile("./digits4000_txt/digits4000_digits_vec.txt")
    DataY = LoadFile("./digits4000_txt/digits4000_digits_labels.txt")
    DataY = np.ravel(DataY)
    TrainX = DataX[:2000, :]
    TrainX /= 256
    TrainY = DataY[:2000]
    TestX = DataX[2000:, :]
    TestX /= 256
    TestY = DataY[2000:]
    TrainYNew = np.zeros((2000,10))
    for i in range(2000):
        TrainYNew[i][int(TrainY[i])] = 1
    TestYNew = np.zeros((2000,10))
    for i in range(2000):
        TestYNew[i][int(TestY[i])] = 1
    
    Dim = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # CNN
    plt.figure(1)
    plt.title("CNN with PCA")
    ACCList = []
    for i in Dim:
        PCATmp = PCA(i)
        PCATmp.fit(TrainX)
        PCATrainX = PCATmp.transform(TrainX)
        PCATestX = PCATmp.transform(TestX)
        model = Sequential()
        model.add(Dense(500,input_shape=(PCATrainX.shape[1],), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation="relu"), )
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(PCATrainX, TrainYNew, batch_size=200, epochs=400, shuffle=True, verbose=0, validation_split=0.1)
        Acc = model.evaluate(PCATestX, TestYNew, batch_size=200, verbose=1)
        print("Default PCA Loss: "+str(Acc[0]))
        print('Default PCA Accuracy:', str(Acc[1]))
        ACCList.append(Acc[1])
    plt.plot(Dim, ACCList, color='green',label='default')
    # CNN with rbf kernel
    ACCList = []
    for i in Dim:
        PCATmp = KernelPCA(n_components=i,kernel="rbf")
        PCATmp.fit(TrainX)
        PCATrainX = PCATmp.transform(TrainX)
        PCATestX = PCATmp.transform(TestX)
        model = Sequential()
        model.add(Dense(500,input_shape=(PCATrainX.shape[1],), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation="relu"), )
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(PCATrainX, TrainYNew, batch_size=200, epochs=400, shuffle=True, verbose=0, validation_split=0.1)
        Acc = model.evaluate(PCATestX, TestYNew, batch_size=200, verbose=1)
        print("Rbf kernel PCA Loss: "+str(Acc[0]))
        print('Rbf kernel PCA Accuracy:', str(Acc[1]))
        ACCList.append(Acc[1])
    plt.plot(Dim, ACCList, color='red',label='rbf kernel')
    # CNN with sigmoid kernel
    ACCList = []
    for i in Dim:
        PCATmp = KernelPCA(n_components=i,kernel="sigmoid")
        PCATmp.fit(TrainX)
        PCATrainX = PCATmp.transform(TrainX)
        PCATestX = PCATmp.transform(TestX)
        model = Sequential()
        model.add(Dense(500,input_shape=(PCATrainX.shape[1],), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation="relu"), )
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(PCATrainX, TrainYNew, batch_size=200, epochs=400, shuffle=True, verbose=0, validation_split=0.1)
        Acc = model.evaluate(PCATestX, TestYNew, batch_size=200, verbose=1)
        print("Sigmoid kernel PCA Loss: "+str(Acc[0]))
        print('Sigmoid kernel PCA Accuracy:', str(Acc[1]))
        ACCList.append(Acc[1])
    plt.plot(Dim, ACCList, color='blue',label='sigmoid kernel')
    plt.xlabel("Dimension")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

    # without using PCA
    # model = Sequential()
    # model.add(Dense(500,input_shape=(TrainX.shape[1],), activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(500, activation="relu"), )
    # model.add(Dropout(0.5))
    # model.add(Dense(10))
    # model.add(Activation("softmax"))
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(TrainX, TrainYNew, batch_size=200, epochs=400, shuffle=True, verbose=0, validation_split=0.1)
    # AccNormal = model.evaluate(TestX, TestYNew, batch_size=200, verbose=1)
    # print("Loss: "+str(AccNormal[0]))
    # print('Accuracy:', str(AccNormal[1]))
    