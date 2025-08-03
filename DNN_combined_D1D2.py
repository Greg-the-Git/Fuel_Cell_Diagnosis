# =================================================================================
#                               IMPORT LIBRARIES
# =================================================================================

import numpy as np
from keras.utils import to_categorical
from network_models import DNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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

# Split data for each class into training and testing sets
drying_x_train, drying_x_test, drying_y_train, drying_y_test = train_test_split(
    drying_v_response, drying_labels, test_size=0.2, random_state=42
)
normal_x_train, normal_x_test, normal_y_train, normal_y_test = train_test_split(
    normal_v_response, normal_labels, test_size=0.2, random_state=42
)
starvation_x_train, starvation_x_test, starvation_y_train, starvation_y_test = (
    train_test_split(
        starvation_v_response, starvation_labels, test_size=0.2, random_state=42
    )
)

new_drying_x_train, new_drying_x_test, new_drying_y_train, new_drying_y_test = (
    train_test_split(
        new_drying_v_response, new_drying_labels, test_size=0.2, random_state=42
    )
)
new_normal_x_train, new_normal_x_test, new_normal_y_train, new_normal_y_test = (
    train_test_split(
        new_normal_v_response, new_normal_labels, test_size=0.2, random_state=42
    )
)
(
    new_starvation_x_train,
    new_starvation_x_test,
    new_starvation_y_train,
    new_starvation_y_test,
) = train_test_split(
    new_starvation_v_response, new_starvation_labels, test_size=0.2, random_state=42
)

# Combine data from different classes for training and testing
x_train = np.vstack(
    (
        drying_x_train,
        normal_x_train,
        starvation_x_train,
        new_drying_x_train,
        new_normal_x_train,
        new_starvation_x_train,
    )
)

y_train = np.hstack(
    (
        drying_y_train,
        normal_y_train,
        starvation_y_train,
        new_drying_y_train,
        new_normal_y_train,
        new_starvation_y_train,
    )
).T

x_test = np.vstack(
    (
        drying_x_test,
        normal_x_test,
        starvation_x_test,
        new_drying_x_test,
        new_normal_x_test,
        new_starvation_x_test,
    )
)

y_test = np.hstack(
    (
        drying_y_test,
        normal_y_test,
        starvation_y_test,
        new_drying_y_test,
        new_normal_y_test,
        new_starvation_y_test,
    )
).T

# One-hot encode the target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# =================================================================================
#                                  TRAIN MODEL
# =================================================================================

# Create an instance of the DNN model
model = DNN()
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
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=80)
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
            color="white" if conf_matrix[i, j] > 40 else "black",
        )

plt.tight_layout()
plt.show()
