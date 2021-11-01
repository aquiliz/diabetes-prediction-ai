from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the data
dataset = loadtxt("diabetes-dataset.csv", delimiter=",")
x = dataset[:, 0:8]
y = dataset[:, 8]

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the Keras model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train (fit) the Keras model
model.fit(x, y, epochs=150, batch_size=10)

# Evaluate the keras model
_, accuracy = model.evaluate(x, y)
print("Accuracy: %.2f" % (accuracy * 100))

# Predict the first 10 cases
predictions = (model.predict(x) > 0.5).astype(int)
for i in range(10):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
