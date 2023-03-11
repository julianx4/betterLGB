from nnfs_model import *

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
confidences = model.predict(data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

print(predictions[0])