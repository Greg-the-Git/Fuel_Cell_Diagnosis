# =============================================================================
#                               IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
from network_models import DNN, CNN
from svm_models import SVM_linear, SVM_rbf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import svm


# =============================================================================
#                                  LOAD DATA
# =============================================================================

# Load drying voltage response data
drying_data = np.loadtxt(
    "data/Processed_Data/Drying_Voltage_Response.csv", delimiter=","
)
drying_v_response = drying_data[:, 0:-1]
drying_labels = drying_data[:, -1]

# Load normal voltage response data
normal_data = np.loadtxt(
    "data/Processed_Data/Normal_Voltage_Response.csv", delimiter=","
)
normal_v_response = normal_data[:, 0:-1]
normal_labels = normal_data[:, -1]

# Load starvation voltage response data
starvation_data = np.loadtxt(
    "data/Processed_Data/Starvation_Voltage_Response.csv", delimiter=","
)
starvation_v_response = starvation_data[:, 0:-1]
starvation_labels = starvation_data[:, -1]

# Load new drying voltage response data
new_drying_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Drying_Voltage_Response.csv", delimiter=","
)
new_drying_v_response = new_drying_data[:, 0:-1]
new_drying_labels = new_drying_data[:, -1]

# Load new normal voltage response data
new_normal_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Normal_Voltage_Response.csv", delimiter=","
)
new_normal_v_response = new_normal_data[:, 0:-1]
new_normal_labels = new_normal_data[:, -1]

# Load new starvation voltage response data
new_starvation_data = np.loadtxt(
    "data/Processed_Data/New_MEA_Starvation_Voltage_Response.csv", delimiter=","
)
new_starvation_v_response = new_starvation_data[:, 0:-1]
new_starvation_labels = new_starvation_data[:, -1]

# =============================================================================
#                                NORMALISE DATA
# =============================================================================

# Normalize the voltage response data for each group
for i in range(drying_v_response.shape[0]):
    drying_v_response[i] -= drying_v_response[i, 0]
    normal_v_response[i] -= normal_v_response[i, 0]
    starvation_v_response[i] -= starvation_v_response[i, 0]
    new_drying_v_response[i] -= new_drying_v_response[i, 0]
    new_normal_v_response[i] -= new_normal_v_response[i, 0]
    new_starvation_v_response[i] -= new_starvation_v_response[i, 0]

# Delete loaded data arrays to free up memory
del (
    drying_data,
    normal_data,
    starvation_data,
    new_drying_data,
    new_normal_data,
    new_starvation_data,
)

# =============================================================================
#                      DIVIDE DATA INTO OLD MEA AND NEW MEA
# =============================================================================

# Combine old MEA data into one array
x_old = np.vstack((drying_v_response, normal_v_response, starvation_v_response))
y_old = np.hstack((drying_labels, normal_labels, starvation_labels)).T

# Combine new MEA data into one array
x_new = np.vstack(
    (new_drying_v_response, new_normal_v_response, new_starvation_v_response)
)
y_new = np.hstack((new_drying_labels, new_normal_labels, new_starvation_labels)).T

# Convert labels to categorical form
y_old = to_categorical(y_old)
y_new = to_categorical(y_new)

# =============================================================================
#                 LOOP OVER DIFFERENT SPLITS OF D2 IN TRAINING DATA
# =============================================================================

# Define an array of splits
splits = np.arange(0, 1, 0.05)

# Initialize lists to store accuracy values
SVM_linear_accs = []
SVM_rbf_accs = []
DNN_stdevs = []
DNN_accs = []
CNN_stdevs = []
CNN_accs = []

# Define the number of iterations for each model training
iterations = 100

# Loop over different splits of D2 in the training data
for split in splits:

    # =============================================================================
    #                 TRAIN-TEST SPLIT BASED ON LOOP ITERATION
    # =============================================================================

    # If split is greater than 0, perform train-test split with new MEA data
    if split > 0:
        x_new_train, x_test, y_new_train, y_test = train_test_split(
            x_new, y_new, train_size=split, random_state=42
        )

        # Combine old and new MEA data for training
        x_train = np.vstack(
            [
                x_old,
                x_new_train,
            ]
        )

        y_train = np.vstack(
            [
                y_old,
                y_new_train,
            ]
        )
    # If split is 0, use only old MEA data for
    # training and new MEA data for testing
    else:
        x_train = x_old
        y_train = y_old
        x_test = x_new
        y_test = y_new

    # =============================================================================
    #                                 TRAIN 1D-CNN
    # =============================================================================

    # Initialize an empty list to store accuracy values for each iteration
    CNN_data = []

    # Loop over the specified number of iterations
    for i in range(iterations):
        # Create a new instance of the CNN model
        CNN_model = CNN()

        # Compile the CNN model
        CNN_model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Train the CNN model on the training data
        history = CNN_model.fit(
            x_train,
            y_train,
            epochs=50,
            validation_split=0.1,
            batch_size=256,
            verbose=False,
        )

        # Evaluate the trained CNN model on the test data
        loss, acc = CNN_model.evaluate(x_test, y_test, batch_size=256, verbose=False)

        # Append the accuracy of the CNN model to the list
        CNN_data.append(acc)
        print(f"Done CNN Split {split} (Iteration {i})")

    # Compute the mean accuracy of the CNN model across all iterations
    CNN_accs.append(np.mean(CNN_data))

    # Compute the standard deviation of the accuracy
    # of the CNN model across all iterations
    CNN_stdevs.append(np.std(CNN_data))

    # Delete the CNN model to free up memory
    del CNN_model

    # =============================================================================
    #                                 TRAIN DNN
    # =============================================================================

    # Initialize an empty list to store accuracy values for each iteration
    DNN_data = []

    # Loop over the specified number of iterations
    for i in range(iterations):
        # Create a new instance of the DNN model
        DNN_model = DNN()

        # Compile the DNN model
        DNN_model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # Train the DNN model on the training data
        history = DNN_model.fit(
            x_train,
            y_train,
            epochs=50,
            validation_split=0.1,
            batch_size=256,
            verbose=False,
        )

        # Evaluate the trained DNN model on the test data
        loss, acc = DNN_model.evaluate(x_test, y_test, batch_size=256, verbose=False)

        # Append the accuracy of the DNN model to the list
        DNN_data.append(acc)
        print(f"Done DNN Split {split} (Iteration {i})")

    # Compute the mean accuracy of the DNN model across all iterations
    DNN_accs.append(np.mean(DNN_data))

    # Compute the standard deviation of the accuracy
    # of the DNN model across all iterations
    DNN_stdevs.append(np.std(DNN_data))

    # Delete the DNN model to free up memory
    del DNN_model

    # =============================================================================
    #                             TRAIN LINEAR SVM
    # =============================================================================

    # Convert categorical labels back to integer labels for SVM
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Initialize SVM classifier with linear kernel
    clf = SVM_linear()

    # Train the SVM classifier on the training data
    clf.fit(x_train, y_train)

    # Predict labels for the test data using the trained SVM classifier
    y_pred = clf.predict(x_test)

    # Compute accuracy of the SVM classifier
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Append the accuracy to the list
    SVM_linear_accs.append(accuracy)

    print(f"Done Linear SVM Split {split}")

    # =============================================================================
    #                               TRAIN RBF SVM
    # =============================================================================

    # Initialize SVM classifier with RBF kernel
    clf = SVM_rbf()

    # Train the SVM classifier on the training data
    clf.fit(x_train, y_train)

    # Predict labels for the test data using the trained SVM classifier
    y_pred = clf.predict(x_test)

    # Compute accuracy of the SVM classifier
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Append the accuracy to the list
    SVM_rbf_accs.append(accuracy)

    print(f"Done RBF SVM Split {split}")

    # Delete the SVM classifier to free up memory
    del clf


# =============================================================================
#                      COLLECT DATA FROM ALL LOOPS AND PLOT
# =============================================================================

# Create a DataFrame to store accuracy values
df = pd.DataFrame(
    {
        "SVM_linear_accs": SVM_linear_accs,
        "SVM_rbf_accs": SVM_rbf_accs,
        "DNN_stdevs": DNN_stdevs,
        "DNN_accs": DNN_accs,
        "CNN_stdevs": CNN_stdevs,
        "CNN_accs": CNN_accs,
    }
)

# Plot the results
plt.plot(splits, df.DNN_accs, "ko-", label="DNN", markersize=4)
plt.plot(splits, df.CNN_accs, "ro-", label="1D-CNN", markersize=4)
plt.plot(splits, df.SVM_linear_accs, "bo-", label="Linear-SVM", markersize=4)
plt.plot(splits, df.SVM_rbf_accs, "go-", label="RBF-SVM", markersize=4)
plt.legend(loc="lower right")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.ylabel("Test Accuracy")
plt.xlabel("Fraction of $D_2$ in Training Set")
plt.minorticks_on()
plt.grid(which="major", linewidth=0.8)
plt.grid(which="minor", linestyle=":", linewidth=0.5)
plt.show()
