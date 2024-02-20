import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

def plot_data(X, y, ax):
    class_0 = X[y.flatten() == 0]
    class_1 = X[y.flatten() == 1]
    ax.scatter(class_0[:, 0], class_0[:, 1], marker='o', label='y=0')
    ax.scatter(class_1[:, 0], class_1[:, 1], marker='x', label='y=1')

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g

def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
        
    for i in range(m):
        z = np.dot(X[i],w)+b
        f_wb_i = sigmoid(z)
        
        cost+=-y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
        
    cost = cost/m
    
    return cost

def compute_gradient_derivatives(X,y,w,b):

    m,n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.0
    z = np.dot(X,w)+b
    f_wb = sigmoid(z)

    err = f_wb - y

    dj_dw = np.dot(X.T,err)/m
    dj_db = np.sum(err)/m

    return dj_dw,dj_db

def compute_gradient(X,y,num_iter,alpha):

    w = np.zeros((X.shape[1],1))
    b = 0.0

    costs = []

    for i in range(num_iter):
        dj_dw,dj_db = compute_gradient_derivatives(X,y,w,b)
        w-=alpha*dj_dw
        b-=alpha*dj_db

        cost = compute_cost_logistic(X,y,w,b)
        costs.append(cost)

        if i %100==0:
            print(f"cost after iteration {i} : {cost}")

    return w,b,costs
    
alpha = 0.01
num_iters = 1000

w,b,costs= compute_gradient(X,y,num_iters,alpha)
print('Optimized parameters:')
print('w_opt:', w)
print('b_opt:', b)
def predict(X, w, b):
    """
    Predicts the class labels for input data X using learned parameters w and b.

    Args:
    X (ndarray): Input data with shape (m, n), where m is the number of examples and n is the number of features.
    w (ndarray): Learned weights with shape (n, 1).
    b (float): Learned bias.

    Returns:
    predictions (ndarray): Predicted class labels (0 or 1) for each example in X.
    """
    # Compute the probabilities using the sigmoid function
    probabilities = sigmoid(np.dot(X, w) + b)
    # Convert probabilities to binary predictions (0 or 1)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions.flatten()

# Assuming you have new data X_new, you can use the predict function like this:
# predictions = predict(X_new, w, b)
# Assuming you have trained the model and obtained the optimized parameters w and b
# Let's create a new example data point
X_new = np.array([[0.5, 0.5]])

# Predict the output for the new example
prediction = predict(X_new, w, b)

print("New example:", X_new)
print("Predicted output:", prediction)


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.axis([0, 4, 0, 3.5])
plot_data(X,y,ax)
plt.show()




