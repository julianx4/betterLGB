from nnfs_model import *
import numpy as np
import os

def load_new_positions():
    datapath = "data/new"
    testfile = "data/new/roundv2-9.npy"
    datafilelist = os.listdir(datapath)

    X=np.array([])
    y=np.array([])
    X_test=np.array([])
    y_test=np.array([])

    np.linspace(0, 1, 26)
    steps = np.linspace(0, 1, 26)

    for file in datafilelist:
        filewithpath = datapath+"/"+file
        count = 0
        arr = np.load(filewithpath, allow_pickle=True)
        print(arr.shape)
        originalshape = arr.shape
        scaled_steps = arr.shape[0] * steps
        for step in scaled_steps[:-1]:
            X = np.append(X, arr[int(step)])
            y = np.append(y, count)

            count += 1

    count = 0
    arr = np.load(testfile, allow_pickle=True)
    print(arr.shape)
    scaled_steps = 0 + (arr.shape[0] - 0) * steps
    for step in scaled_steps[:-1]:
        X_test = np.append(X, arr[int(step)])
        y_test = np.append(y, count)
        count += 1

    X = X.reshape((len(X)//originalshape[1]),originalshape[1])
    X_test = X_test.reshape((len(X_test)//originalshape[1]),originalshape[1])

    return np.array(X), np.array(y).astype('uint8'), np.array(X_test), np.array(y_test).astype('uint8')



X, y, X_test, y_test = load_new_positions()


keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

model = Model()

model.add(Layer_Dense(X.shape[1], 60))
model.add(Activation_ReLU())
model.add(Layer_Dense(60, 60))
model.add(Activation_ReLU())
model.add(Layer_Dense(60, 26))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=5e-4), accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=450, batch_size=None, print_every=1000)

model.save('/models/train_model_classification.model')

for array in X_test:
    confidences = model.predict(array)
    predictions = model.output_layer_activation.predictions(confidences)
    print(predictions, confidences[predictions])


testfile = "data/new/roundv2-9.npy"
arr = np.load(testfile, allow_pickle=True)
for array in arr:
    print(array)