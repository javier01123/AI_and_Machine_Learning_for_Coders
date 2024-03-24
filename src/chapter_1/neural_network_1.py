import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

layer0 = Dense(units=1, input_shape=[1])
model = Sequential([layer0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

#make 1 demo prediction 
input_data = np.array([[10.0]]) # Reshaped to shape (1, 1)
prediction = model.predict(input_data)
print(model.predict(input_data))
# result Y = 2X -1  = 19

#result
print("Here is what I learned: {}".format(layer0.get_weights()))
#Thus, the learned relationship between X and Y was Y = 1.9967953X – 0.9900647.