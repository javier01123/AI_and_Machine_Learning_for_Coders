import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28,28)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
# ])

###alternative to prevent warnings fron passing in 'input_shape'
inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
###

model.compile(optimizer = 'adam',              
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=4)

print('EVALUATION:')
model.evaluate(test_images, test_labels)

print('CLASSIFICATIONS:')
#These are the probabilities
#that the image matches the label at that particular index. So, what the neural network
#is reporting is that there’s a 91.4% chance that the item of clothing at index 0 is label 9.
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])