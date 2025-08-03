# =================================================================================
#                               IMPORT LIBRARIES
# =================================================================================

import numpy as np
from svm_models import SVM_rbf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

# Combine data and labels from all classes for training and testing
x_test = np.vstack((drying_v_response, normal_v_response, starvation_v_response))
y_test = np.hstack((drying_labels, normal_labels, starvation_labels)).T
x_train = np.vstack(
    (new_drying_v_response, new_normal_v_response, new_starvation_v_response)
)
y_train = np.hstack((new_drying_labels, new_normal_labels, new_starvation_labels)).T

# =================================================================================
#                                  TRAIN MODEL
# =================================================================================

# Create an instance of the SVM_rbf model
clf = SVM_rbf()

# Fit the model to the training data
clf.fit(x_train, y_train)

# =================================================================================
#                                CONFUSION MATRIX
# =================================================================================

# Generate predictions for the test set
y_pred = clf.predict(x_test)

# Compute accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

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
