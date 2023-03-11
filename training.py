from nnfs_model import *
import numpy as np

position1_file="data/position1.npy"
position2_file="data/position2.npy"
position3_file="data/position3.npy"
position4_file="data/position4.npy"
position5_file="data/position5.npy"
position6_file="data/position6.npy"

def load_position_data():

    # Create lists for samples and labels
    X = []
    y = []
    X_test = []
    y_test = []

    arr1 = np.load(position1_file, allow_pickle=True)
    arr1 = arr1.flatten()
    arr2 = np.load(position2_file, allow_pickle=True)
    arr2 = arr2.flatten()
    arr3 = np.load(position3_file, allow_pickle=True)
    arr3 = arr3.flatten()
    arr4 = np.load(position4_file, allow_pickle=True)
    arr4 = arr4.flatten()
    arr5 = np.load(position5_file, allow_pickle=True)
    arr5 = arr5.flatten()
    arr6 = np.load(position6_file, allow_pickle=True)
    arr6 = arr6.flatten()

    for i in range(0, len(arr1)//5):
        y.append(0)
    for i in range(0, len(arr2)//5):
        y.append(1)
    for i in range(0, len(arr3)//5):
        y.append(2)
    for i in range(0, len(arr4)//5):
        y.append(3)    
    for i in range(0, len(arr5)//5):
        y.append(4)
    for i in range(0, len(arr6)//5):
        y.append(5)
    
    X = np.append(arr1, arr2)
    X = np.append(X, arr3)
    X = np.append(X, arr4)
    X = np.append(X, arr5)
    X = np.append(X, arr6)

    X = X.reshape((len(X)//5),5)

    training_data_keys = [109, 103, 85, 78, 69, 63, 50, 45, 29, 20, 8, 3]

    for key in training_data_keys:
        X_test = np.append(X_test, X[key])
        y_test = np.append(y_test, y[key])
    for key in training_data_keys:
        X = np.delete(X, key, 0)
        y = np.delete(y, key)

    X_test= np.array(X_test).reshape((len(X_test)//5),5)
    return np.array(X), np.array(y).astype('uint8'), np.array(X_test), np.array(y_test).astype('uint8')

X, y, X_test, y_test= load_position_data()


keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

model = Model()

model.add(Layer_Dense(X.shape[1], 100))
model.add(Activation_ReLU())
model.add(Layer_Dense(100, 6))
model.add(Activation_Softmax())

model.set( loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=5e-4), accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=1000, batch_size=None, print_every=1000)

model.save('train_model_test.model')