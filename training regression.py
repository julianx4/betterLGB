from nnfs_model import *
import numpy as np
import os

def load_position_data():

    # Create lists for samples and labels
    X = []
    y = []
    X_test = []
    y_test = []

    datapath = "data/new"
    datafilelist = os.listdir(datapath)
    print("bla")
    for file in datafilelist:
        filewithpath = datapath+"/"+file
        arr = np.load(filewithpath, allow_pickle=True)
        originalshape = arr.shape
        
        
        for i in range(arr.shape[0]):
            y.append(i / (len(arr)/arr.shape[1]))

        arr = arr.flatten()
        X = np.append(X, arr)
    
    X = X.reshape((len(X)//originalshape[1]),originalshape[1])
    #X_test = np.append(X_test, arr10)

    #X = X.reshape((len(X)//5),5)
    #X_test = X_test.reshape((len(X_test)//5),5)

    return np.array(X), np.array(y) #, np.array(X_test), np.array(y_test)

X, y = load_position_data()
X_test = X
y_test = y
print(X.shape)
print(y.shape)

#keys = np.array(range(X.shape[0]))
#np.random.shuffle(keys)
#X = X[keys]
#y = y[keys]

y = y.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#X_test = X_test[:,[1,2,3,4]]
#X = X[:,[1,2,3,4]]

#print(X)
#print(y)

#print(X.shape)
#print(y.shape)

model = Model()

model.add(Layer_Dense(X.shape[1], 100))
model.add(Activation_Linear())
model.add(Layer_Dense(100, 100))
model.add(Activation_Linear())
model.add(Layer_Dense(100, 1))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.005,decay=1e-3), accuracy=Accuracy_Regression())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=1000, batch_size=None, print_every=100)

model.save('train_model_regression.model')

"""
for array in X_test:
    confidences = model.predict(array)
    predictions = model.output_layer_activation.predictions(confidences)
    print(predictions[0][0])

"""