###### Your ID ######
# ID1: 208000158
# ID2: 315390252
#####################

# imports 
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X-X.mean(axis=0)) / (X.max(axis=0)-X.min(axis=0))
    y = (y-y.mean(axis=0)) / (y.max(axis=0)-y.min(axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    # if X.ndim == 1:
    #   X = X.reshape(-1, 1)
    # ones = np.ones((X.shape[0], 1))
    # # Concatenate the ones column to the left of X
    # X = np.concatenate([ones, X], axis=1)
    X = np.column_stack((np.ones(X.shape[0]), X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = len(y)  # number of training examples
    predictions = X.dot(theta)         # Compute predictions h_theta(x) = X * theta
    errors = predictions - y           # Compute errors for each instance
    J = (1 / (2 * m)) * np.sum(errors ** 2)  # Mean Squared Error cost function
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()  # Work on a copy of theta
    J_history = []        # To keep track of the cost over iterations
    m = len(y)            # Number of training examples

    for iteration in range(num_iters):
        preds = X @ theta               # Compute predictions
        error = preds - y               # Calculate error for each training example
        grad = (error @ X) / m          # Compute the gradient vector
        theta = theta - alpha * grad    # Update theta using the gradient
        cost = compute_cost(X, y, theta.copy())  # Evaluate the cost with the updated theta
        J_history.append(cost)          # Record the cost

    return theta, J_history   


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    X_T = X.T
    # Compute the inverse of (X^T X) and then multiply by X^T and y.
    pinv_theta = np.linalg.inv(X_T @ X) @ X_T @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

import numpy as np

def efficient_gradient_descent(X, y, theta, alpha, iterations):
    """
    Learn the parameters of your model using the *training set*, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()
    J_history = []

    # Compute initial cost
    prev_J = compute_cost(X, y, theta)

    i = 0
    while i < iterations:
        # Gradient step
        gradient = (X.T @ (X @ theta - y)) / y.shape[0]
        theta -= alpha * gradient

        # Compute and record new cost _after_ the update
        current_J = compute_cost(X, y, theta)
        J_history.append(current_J)

        # Safety check
        if np.isnan(current_J) or np.isinf(current_J):
            break

        # Early stop if improvement is tiny
        if abs(prev_J - current_J) < 1e-8:
            break

        prev_J = current_J
        i += 1

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over provided values of alpha and train a model using the 
    *training* dataset. maintain a python dictionary with alpha as the 
    key and the loss on the *validation* set as the value.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {key (alpha) : value (validation loss)}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(42)
    init_theta = np.random.random(size=2)
    for alpha in alphas:
        thetas = gradient_descent(X_train, y_train, init_theta, alpha, iterations)[0]
        alpha_dict[alpha] = compute_cost(X_val, y_val, thetas)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    remaining_features = set(range(1, X_train.shape[1]))

    for _ in range(5):
        candidate_costs = {}
        for feat in remaining_features:
            current_feats = selected_features + [feat]
            np.random.seed(42)
            theta = np.random.random(len(current_feats))
            cost = compute_cost(
                X_val[:, current_feats],
                y_val,
                efficient_gradient_descent(X_train[:, current_feats], y_train, theta, best_alpha, iterations)[0]
            )
            candidate_costs[feat] = cost
        best_feat = min(candidate_costs, key=candidate_costs.get)
        selected_features.append(best_feat)
        remaining_features.remove(best_feat)

    selected_features = [feat - 1 for feat in selected_features]
    return selected_features



def create_square_features(df):
    from itertools import combinations_with_replacement
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    feature_names = df.columns
    new_features = {}

    for feat1, feat2 in combinations_with_replacement(feature_names, 2):
        if feat1 == feat2:
            new_col_name = f"{feat1}^2"
            new_features[new_col_name] = df[feat1] ** 2
        else:
            new_col_name = f"{feat1}*{feat2}"
            new_features[new_col_name] = df[feat1] * df[feat2]

    # Use pd.concat to add all new features at once
    df_poly = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    return df_poly