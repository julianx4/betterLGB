import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


round1_file="data/round1.npy"
round2_file="data/round2.npy"
round3_file="data/round3.npy"
round4_file="data/round4.npy"
round5_file="data/round5.npy"
round6_file="data/round6.npy"
round7_file="data/round7.npy"
round8_file="data/round8.npy"
round9_file="data/round9.npy"

def load_position_data():

    # Create lists for samples and labels
    X = []
    y = []
    X_test = []
    y_test = []

    arr1 = np.load(round1_file, allow_pickle=True)
    arr1 = arr1.flatten()
    arr2 = np.load(round2_file, allow_pickle=True)
    arr2 = arr2.flatten()
    arr3 = np.load(round3_file, allow_pickle=True)
    arr3 = arr3.flatten()
    arr4 = np.load(round4_file, allow_pickle=True)
    arr4 = arr4.flatten()
    arr5 = np.load(round5_file, allow_pickle=True)
    arr5 = arr5.flatten()
    arr6 = np.load(round6_file, allow_pickle=True)
    arr6 = arr6.flatten()
    arr7 = np.load(round7_file, allow_pickle=True)
    arr7 = arr7.flatten()
    arr8 = np.load(round8_file, allow_pickle=True)
    arr8 = arr8.flatten()
    arr9 = np.load(round9_file, allow_pickle=True)
    arr9 = arr9.flatten()

    for i in range(0, len(arr1)//5):
        y.append(i / (len(arr1)/5))
    for i in range(0, len(arr2)//5):
        y.append(i / (len(arr2)/5))
    for i in range(0, len(arr3)//5):
        y.append(i / (len(arr3)/5))
    for i in range(0, len(arr4)//5):
        y.append(i / (len(arr4)/5))    
    for i in range(0, len(arr5)//5):
        y.append(i / (len(arr5)/5))
    for i in range(0, len(arr6)//5):
        y.append(i / (len(arr6)/5))
    for i in range(0, len(arr7)//5):
        y.append(i / (len(arr7)/5))
    for i in range(0, len(arr8)//5):
        y.append(i / (len(arr8)/5))
    for i in range(0, len(arr9)//5):
        y.append(i / (len(arr9)/5))

    X = np.append(X, arr1)
    X = np.append(X, arr2)
    X = np.append(X, arr3)
    X = np.append(X, arr4)
    X = np.append(X, arr5)
    X = np.append(X, arr6)
    X = np.append(X, arr7)
    X = np.append(X, arr8)
    X = np.append(X, arr9)

    X_test = np.append(X_test, arr9)

    X = X.reshape((len(X)//5),5)
    X_test = X_test.reshape((len(X_test)//5),5)

    return np.array(X), np.array(y), np.array(X_test), np.array(y_test)

X, y, X_test, y_test= load_position_data()
"""
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys].
"""
y_train= y.reshape(-1,1)
x_train = X[:,[2,3,4]]

dataset_numpy = np.append(y_train, x_train, axis=1)

dataset = pd.DataFrame(dataset_numpy, columns=["position", "x", "y", "z"])
print(dataset.tail())


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(dataset[["position", "x", "y", "z"]], diag_kind='kde')
#plt.show()

print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('position')
test_labels = test_features.pop('position')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.summary()

linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.show()

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)