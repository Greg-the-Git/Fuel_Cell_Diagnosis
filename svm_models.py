from sklearn import svm


def SVM_linear():
    """
    Creates a Support Vector Machine (SVM) classifier with a linear kernel.

    Returns:
    svm.SVC: SVM classifier with linear kernel.
    """
    return svm.SVC(kernel="linear")


def SVM_rbf():
    """
    Creates a Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel.

    Returns:
    svm.SVC: SVM classifier with RBF kernel.
    """
    return svm.SVC(kernel="rbf")
