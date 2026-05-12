import math
import numpy as np
import matplotlib.pyplot as plt

#used functions
def sigmoid(z):
 g = 1 / (1 + np.exp(-z))
 return g

def cost_function (X,y,w,b):
 m,n = X.shape
 loss_sum = 0

 for i in range(m):
  z_wb = 0
  for j in range(n):
   z_wb_ij = w[j]* X[i][j]
   z_wb += z_wb_ij

  z_wb = z_wb + b
  f_wb = sigmoid(z_wb)
  loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb) # use np.log
  loss_sum += loss

 total_sum = (1 / m) * loss_sum
 return total_sum

def gradient_function(X,y,w,b):
 m, n = X.shape
 dj_dw = np.zeros(w.shape)
 dj_db = 0.

 for i in range(m):
  z_wb = 0

  for j in range(n):
   z_wb_ij = w[j]* X[i][j]
   z_wb += z_wb_ij

  z_wb += b
  f_wb = sigmoid(z_wb)

  dj_db_i = f_wb - y[i]
  dj_db += dj_db_i

  for j in range(n):
   dj_dw_ij = (f_wb - y[i]) * X[i][j]
   dj_dw[j] += dj_dw_ij

 dj_dw = dj_dw / m
 dj_db = dj_db / m
 return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_inters, lambda_):
 m = len(X)
 J_history = []
 w_history = []

 for i in range(num_inters):
  dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
  w_in = w_in - alpha * dj_dw
  b_in = b_in - alpha * dj_db

  if i < 100000:
   cost = cost_function(X, y, w_in, b_in, lambda_)
   J_history.append(cost)

  if i% math.ceil(num_inters/10) == 0 or i == (num_inters - 1):
   w_history.append(w_in)
   print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

 return w_in, b_in, J_history, w_history

def plot_decision_boundary(w, b, X_original, y):
    #plot original 2D data
    plot_data(X_original, y, pos_label="Accepted", neg_label="Rejected")

    #create grid for original x1 and x2 space
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))

    # Evaluate model on the grid
    for i in range(len(u)):
        for j in range(len(v)):
            mapped_features = map_feature(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = (np.dot(mapped_features, w) + b).item()

    z = z.T

    #draw line where z = 0
    plt.contour(u, v, z, levels=[0], linewidths=2)

#data set
X_train = np.array([
    [0.051267, 0.69956],
    [-0.092742, 0.68494],
    [-0.21371, 0.69225],
    [-0.375, 0.50219],
    [-0.51325, 0.46564],
    [-0.52477, 0.2098],
    [-0.39804, 0.034357],
    [-0.30588, -0.19225],
    [0.016705, -0.40424],
    [0.13191, -0.51389],
    [0.38537, -0.56506],
    [0.52938, -0.5212],
    [0.63882, -0.24342],
    [0.73675, -0.18494],
    [0.54666, 0.48757],
    [0.322, 0.5826],
    [0.16647, 0.53874],
    [-0.046659, 0.81652],
    [-0.17339, 0.69956],
    [-0.47869, 0.63377],
    [-0.60541, 0.59722],
    [-0.62846, 0.33406],
    [-0.59389, 0.005117],
    [-0.42108, -0.27266],
    [-0.11578, -0.39693],
    [0.20104, -0.60161],
    [0.46601, -0.53582],
    [0.67339, -0.53582],
    [-0.13882, 0.54605],
    [-0.29435, 0.77997],
    [-0.26555, 0.96272],
    [-0.16187, 0.8019],
    [-0.17339, 0.64839],
    [-0.28283, 0.47295],
    [-0.36348, 0.31213],
    [-0.30012, 0.027047],
    [-0.23675, -0.21418],
    [-0.06394, -0.18494],
    [0.062788, -0.16301],
    [0.22984, -0.41155],
    [0.2932, -0.2288],
    [0.48329, -0.18494],
    [0.64459, -0.14108],
    [0.46025, 0.012427],
    [0.6273, 0.15863],
    [0.57546, 0.26827],
    [0.72523, 0.44371],
    [0.22408, 0.52412],
    [0.44297, 0.67032],
    [0.322, 0.69225],
    [0.13767, 0.57529],
    [-0.0063364, 0.39985],
    [-0.092742, 0.55336],
    [-0.20795, 0.35599],
    [-0.20795, 0.17325],
    [-0.43836, 0.21711],
    [-0.21947, -0.016813],
    [-0.13882, -0.27266],
    [0.18376, 0.93348],
    [0.22408, 0.77997],
    [0.29896, 0.61915],
    [0.50634, 0.75804],
    [0.61578, 0.7288],
    [0.60426, 0.59722],
    [0.76555, 0.50219],
    [0.92684, 0.3633],
    [0.82316, 0.27558],
    [0.96141, 0.085526],
    [0.93836, 0.012427],
    [0.86348, -0.082602],
    [0.89804, -0.20687],
    [0.85196, -0.36769],
    [0.82892, -0.5212],
    [0.79435, -0.55775],
    [0.59274, -0.7405],
    [0.51786, -0.5943],
    [0.46601, -0.41886],
    [0.35081, -0.57968],
    [0.28744, -0.76974],
    [0.085829, -0.75512],
    [0.14919, -0.57968],
    [-0.13306, -0.4481],
    [-0.40956, -0.41155],
    [-0.39228, -0.25804],
    [-0.74366, -0.25804],
    [-0.69758, 0.041667],
    [-0.75518, 0.2902],
    [-0.69758, 0.68494],
    [-0.4038, 0.70687],
    [-0.38076, 0.91886],
    [-0.50749, 0.90424],
    [-0.54781, 0.70687],
    [0.10311, 0.77997],
    [0.057028, 0.91886],
    [-0.10426, 0.99196],
    [-0.081221, 1.1089],
    [0.28744, 1.087],
    [0.39689, 0.82383],
    [0.63882, 0.88962],
    [0.82316, 0.66301],
    [0.67339, 0.64108],
    [1.0709, 0.10015],
    [-0.046659, -0.57968],
    [-0.23675, -0.63816],
    [-0.15035, -0.36769],
    [-0.49021, -0.3019],
    [-0.46717, -0.13377],
    [-0.28859, -0.060673],
    [-0.61118, -0.067982],
    [-0.66302, -0.21418],
    [-0.59965, -0.41886],
    [-0.72638, -0.082602],
    [-0.83007, 0.31213],
    [-0.72062, 0.53874],
    [-0.59389, 0.49488],
    [-0.48445, 0.99927],
    [-0.0063364, 0.99927],
    [0.63265, -0.030612],
])

y_train = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
])

print("Training examples" + "\nShape of y_train is: " + str(y_train.shape) + "\nShape of X_train is: " + str(X_train.shape))

#feature mapping function to polynomial features, just for experiment
def map_feature(X1, X2, degree=6):
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            feature = (X1 ** (i - j)) * (X2 ** j)
            out.append(feature)
    return np.stack(out, axis=1)

# Plot examples
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
 positive = y == 1
 negative = y == 0
 plt.scatter(X[positive, 0], X[positive, 1], marker="x", label=pos_label)
 plt.scatter(X[negative, 0], X[negative, 1], marker="o", label=neg_label)
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

#show training set after feature
print("Original shape of data:", X_train.shape)
mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

#here we used lambda regularization to reduce polynomial features impact for w because it overfit
#lambda set a "penalty" for extra big weight, and make plot smooth and dont delete feature at all
#less lambda = less penalty, big = big penalty
#b isn't any scence to regulize,but it can be done also
def cost_function_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    #calls the cost_function_reg
    cost_without_reg = cost_function(X, y, w, b)
    #need to calculate this value
    reg_cost = 0.

    for j in range(n):
        reg_cost_j = w[j] ** 2
        reg_cost += reg_cost_j

    reg_cost = (lambda_ / (2 * m)) * reg_cost
    #add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = cost_function_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost) #can be shown, should be 0.6618252552483948


def gradient_function_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = gradient_function(X, y, w, b)

    for j in range(n):
        dj_dw_j_reg = (lambda_ / m) * w[j]
        dj_dw[j] += dj_dw_j_reg

    return dj_db, dj_dw

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
dj_db, dj_dw = gradient_function_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", ) #Can be shown, should be 0.07138288792343662
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", ) #Can be shown, should be [-0.010386028450548701, 0.011409852883280124, 0.0536273463274574, 0.003140278267313462]

#lets test our function with reg
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.
#set regularization parameter lambda_
lambda_ = 0.01
#gradient descent settings
iterations = 10000
alpha = 0.01
w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    cost_function_reg, gradient_function_reg,
                                    alpha, iterations, lambda_)
#show plot
plot_decision_boundary(w, b, X_train, y_train)
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show() #can be shown