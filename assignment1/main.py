"""
Problem 1:
(a) [35 points] Plot the two errors against the corresponding k values. Make sure that you
annotate the plots (e.g. using different colors or types of line) and add legends as needed.
(b) [15 points] Does the plot match (approximately) your intuition of the bias-variance tradeoff in terms of having a U-shaped error? What is the optimal k value based on this result?
For the optimal k, what are the corresponding degrees-of-freedom and its error?

Problem 2:
(a) [5 points] Generate 4 independent standard Normal variables X1, X2, X3, X4 of n = 1000
independent observations. You can then generate a response Y as follows:
Y = X1 + 2 · X2 - X3 + ε
with IID N(0, 1) errors ε. Set the random seed to 598 for reproducibility.
(b) [10 points] Use the appropriate kNN function in R or Python and report the mean squared
error (MSE) for your prediction with k = 4. Use the first 500 observations as the training
data and the rest as testing data. Predict the response using the built-in kNN function
with k = 5.
(c) [35 points] For this question, you cannot load any additional packages. Write your own
kNN function, mykNN (xtrain, ytrain, xtest, k), that fits a kNN model and predicts
multiple target points xtest. The function should return a variable ytest.
Notes:
    - xtrain is the training data set feature
    - ytrain is the training data set response
    - xtest is the testing data set feature
    - ytest is the testing data set prediction
"""

from ucimlrepo import fetch_ucirepo 

def plot_errors_vs_k(k_values, train_errors, test_errors):
    """
    Plots training and testing errors against k values.
    """
    pass  # TODO: Implement plotting logic

def generate_data():
    """
    Generates X1, X2, X3, X4 (standard normal) and Y as specified.
    Returns X (n x 4), Y (n,)
    """  

    # code from the UCI Data Repository: https://doi.org/10.24432/C5MG6K
    # fetch dataset 
    pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81) 
    
    # data (as pandas dataframes) 
    X = pen_based_recognition_of_handwritten_digits.data.features 
    y = pen_based_recognition_of_handwritten_digits.data.targets 
    
    # metadata 
    # print(pen_based_recognition_of_handwritten_digits.metadata)
    
    # variable information 
    # print(pen_based_recognition_of_handwritten_digits.variables)


def knn_predict_builtin(xtrain, ytrain, xtest, k):
    """
    Uses a built-in kNN function to predict y for xtest.
    Returns ytest_pred (predictions for xtest).
    """
    pass  # TODO: Implement using sklearn or similar

def mean_squared_error(y_true, y_pred):
    """
    Computes mean squared error between y_true and y_pred.
    """
    pass  # TODO: Implement MSE calculation

def mykNN(xtrain, ytrain, xtest, k):
    """
    Custom implementation of kNN regression.
    xtrain: (n_train, d)
    ytrain: (n_train,)
    xtest: (n_test, d)
    Returns ytest_pred: (n_test,)
    """
    pass  # TODO: Implement custom kNN

if __name__ == "__main__":
    # Example usage and workflow
    # 1. Generate data
    generate_data()
    # 2. Split into train/test
    # 3. Use built-in kNN and custom kNN
    # 4. Plot errors vs k
    pass  # TODO: