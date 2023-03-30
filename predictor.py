from nnfs_model import *

# Load the model
model = Model.load('models/train_model_classification.model')

model = Model()
model.add(Layer_Dense(12, 60))
model.add(Activation_ReLU())
model.add(Layer_Dense(60, 60))
model.add(Activation_ReLU())
model.add(Layer_Dense(60, 26))
model.add(Activation_Softmax())
model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=5e-4), accuracy=Accuracy_Categorical())
model.finalize()


# Predict on the image
confidences = model.predict([1,1,2,1,1,1,1,1,1,1,1,1])

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

print(predictions[0])