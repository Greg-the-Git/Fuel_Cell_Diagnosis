# =================================================================================
#                               IMPORT LIBRARIES
# =================================================================================

import numpy as np
from network_models import CNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

# =================================================================================
#                                  LOAD DATA
# =================================================================================

# Load data for each class
drying_data = np.loadtxt(
    "data/Processed_Data/Drying_Voltage_Response.csv", delimiter=","
)
drying_v_response = drying_data[:, 0:-1]
drying_labels = drying_data[:, -1]

normal_data = np.loadtxt(
    "data/Processed_Data/Normal_Voltage_Response.csv", delimiter=","
)
normal_v_response = normal_data[:, 0:-1]
normal_labels = normal_data[:, -1]

starvation_data = np.loadtxt(
    "data/Processed_Data/Starvation_Voltage_Response.csv", delimiter=","
)
starvation_v_response = starvation_data[:, 0:-1]
starvation_labels = starvation_data[:, -1]

new_drying_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Drying_Voltage_Response.csv", delimiter=","
)
new_drying_v_response = new_drying_data[:, 0:-1]
new_drying_labels = new_drying_data[:, -1]

new_normal_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Normal_Voltage_Response.csv", delimiter=","
)
new_normal_v_response = new_normal_data[:, 0:-1]
new_normal_labels = new_normal_data[:, -1]

new_starvation_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Starvation_Voltage_Response.csv", delimiter=","
)
new_starvation_v_response = new_starvation_data[:, 0:-1]
new_starvation_labels = new_starvation_data[:, -1]

# =================================================================================
#                                NORMALISE DATA
# =================================================================================

# Normalize the data for each class
for data in [
    drying_v_response,
    normal_v_response,
    starvation_v_response,
    new_drying_v_response,
    new_normal_v_response,
    new_starvation_v_response,
]:
    for i in range(data.shape[0]):
        data[i] -= data[i, 0]

# =================================================================================
#                                TRAIN-TEST SPLIT
# =================================================================================

# Combine data from different classes for training and testing
x_train = np.vstack((drying_v_response, normal_v_response, starvation_v_response))

# Combine labels from different classes for training
y_train = np.hstack((drying_labels, normal_labels, starvation_labels)).T

# One-hot encode the target labels for training
y_train = to_categorical(y_train)

# Combine data from different classes for testing
x_test = np.vstack(
    (new_drying_v_response, new_normal_v_response, new_starvation_v_response)
)

# Combine labels from different classes for testing
y_test = np.hstack((new_drying_labels, new_normal_labels, new_starvation_labels)).T

# One-hot encode the target labels for testing
y_test = to_categorical(y_test)

# =================================================================================
#                                  TRAIN MODEL
# =================================================================================

# Create an instance of the CNN model
model = CNN()
model.summary()

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=256)

# Evaluate the model on the test set
loss, acc = model.evaluate(x_test, y_test, batch_size=256)

# =================================================================================
#                                CONFUSION MATRIX
# =================================================================================

# Generate predictions for the test set
y_pred = np.argmax(model.predict(x_test, batch_size=20), axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

# Plot confusion matrix
plt.figure()
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=200)
plt.colorbar()

classes = [
    "Drying",
    "Normal",
    "Starvation",
]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicted labels")
plt.ylabel("True labels")

# Add text annotations
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(
            j,
            i,
            format(conf_matrix[i, j]),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > 100 else "black",
        )

plt.tight_layout()
plt.show()
