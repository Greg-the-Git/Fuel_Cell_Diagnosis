# =================================================================================
#                               IMPORT LIBRARIES
# =================================================================================

import numpy as np
from sklearn.model_selection import train_test_split
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

# =================================================================================
#                                NORMALISE DATA
# =================================================================================

# Normalize the data for each class
for data in [drying_v_response, normal_v_response, starvation_v_response]:
    for i in range(data.shape[0]):
        data[i] -= data[i, 0]

# =================================================================================
#                                TRAIN-TEST SPLIT
# =================================================================================

# Split data into training and testing sets for each class
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

# Combine the training and testing data and labels from all classes
x_train = np.vstack((drying_x_train, normal_x_train, starvation_x_train))
y_train = np.hstack((drying_y_train, normal_y_train, starvation_y_train)).T
x_test = np.vstack((drying_x_test, normal_x_test, starvation_x_test))
y_test = np.hstack((drying_y_test, normal_y_test, starvation_y_test)).T

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
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=40)
plt.colorbar(label="Predictions")

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
            color="white" if conf_matrix[i, j] > 20 else "black",
        )

plt.tight_layout()
plt.show()
