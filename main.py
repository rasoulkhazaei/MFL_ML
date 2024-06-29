#%% input file: test1.TXT
import matplotlib.pyplot as plt
import numpy as np
filename = "test1.TXT"
data = np.loadtxt(filename)

axcalibr = data[1800:1870, 1:2]
radcalibr = data[1800:1870, 2:3]


##########################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Step 1: Preprocess the signal (example)
def preprocess_signal(signal):
    # Normalize the signal
    return (signal - np.mean(signal)) / np.std(signal)

# Step 2: Define the base shapes (kernels)
def create_base_shapes():
    # Example base shapes (e.g., sine wave, square wave, etc.)
    base_shapes = np.array([
        axcalibr
        # np.sin(np.linspace(0, 2 * np.pi, 100)),
        # np.sign(np.sin(np.linspace(0, 2 * np.pi, 100)))  # Square wave
    ])
    return base_shapes

# Step 3: Design the neural network
def build_model(input_length, base_shapes):
    input_signal = Input(shape=(input_length, 1))

    # Create a convolutional layer with the base shapes as kernels
    conv_layer = Conv1D(
        filters=len(base_shapes), 
        kernel_size=base_shapes.shape[1], 
        weights=base_shapes, 
        use_bias=False, 
        trainable=False,
        padding='same'
    )(input_signal)

    # Flatten and add an output layer
    flat_layer = Flatten()(conv_layer)
    output_layer = Dense(225293, activation='linear')(flat_layer)

    model = Model(inputs=input_signal, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse')
    return model
######################################################
input_length = len(data[:, 1])
print(np.array(data[:, 1:2]).shape)
base_shapes = create_base_shapes()

model = build_model(input_length, base_shapes)
labels = np.random.randn(input_length, 1)  # Dummy labels
model.fit(np.array(data[:, 1:2]), labels, epochs=10)

bottleneck_output = Model(inputs=model.input, outputs=model.layers[1].output)
bottleneck_values = bottleneck_output.predict(data[:, 1])



plt.plot(data[:, 1])
plt.plot(bottleneck_values)
plt.show()
