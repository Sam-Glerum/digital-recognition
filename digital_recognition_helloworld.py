import tensorflow as tf

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Convert the ints to floats
x_train, x_test = x_train / 255.0, x_test / 255.0

#define training model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#setup predictions
predictions = model(x_train[:1]).numpy()
predictions