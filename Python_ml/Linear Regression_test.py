import numpy as np
import matplotlib.pylab as plt
from utils import *
import copy
import math

#задаем входные параметры x - как тренинг примеры, y - как результат этих примеров

x_train = np.array([
        20, 22, 25, 27, 30, 32, 35, 37, 40, 42,
        45, 47, 50, 52, 55, 57, 60, 62, 65, 67,
        70, 72, 75, 77, 80, 82, 85, 87, 90, 92,
        95, 97, 21, 24, 28, 31, 34, 38, 41, 44,
        48, 51, 54, 58, 61, 64, 68, 71, 74, 78,
        81, 84, 88, 91, 94, 96, 23, 26, 29, 33,
        36, 39, 43, 46, 49, 53, 56, 59, 63, 66,
        69, 73, 76, 79, 83, 86, 89, 93, 19, 18,
        17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0, 50, 75
    ])

y_train = np.array([
       0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.28,
        0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.52, 0.55, 0.60, 0.62,
        0.65, 0.68, 0.72, 0.75, 0.78, 0.80, 0.85, 0.88, 0.92, 0.94,
        0.96, 0.98, 0.06, 0.09, 0.11, 0.14, 0.17, 0.22, 0.26, 0.29,
        0.33, 0.41, 0.44, 0.50, 0.53, 0.58, 0.61, 0.66, 0.70, 0.76,
        0.79, 0.83, 0.87, 0.91, 0.95, 0.97, 0.07, 0.09, 0.13, 0.16,
        0.19, 0.23, 0.27, 0.31, 0.34, 0.43, 0.47, 0.51, 0.56, 0.59,
        0.63, 0.67, 0.71, 0.74, 0.81, 0.84, 0.89, 0.93, 0.04, 0.03,
        0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01,0.02, 0.01, 
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.40, 0.73
    ])

#делаем z-scale нормализацию для х[2] что бы не улетело

x_train = (x_train - x_train.mean()) / x_train.std()

#делаем инженерию признаков, добавляем дополнительный x[2] что бы входные данные были [x,x[2]]

X_train = np.array([[x, x ** 2]for x in x_train])

#функция для рассчета коста, стоимости J (cost)

def compute_cost(x,y,w,b):

    m = x.shape[0]
    cost= 0
    total_cost = 0

    for i in range(m):
        f_wb = np.dot(x[i], w) + b #используем функцию дот так как она быстрее чем цикл
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m)) * cost

    return total_cost

initial_w = np.array([1,1])
initial_b = 0.1
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print(f"Cost function J is: {cost:.4f}" )

#функция для рассчета градиента (gradient decent)

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

#initial_w = np.array([0,0])
#initial_b = 0
#tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, initial_w, initial_w)
#print(f"Gradient at zero w, b : {tmp_dj_dw:.4f} {tmp_dj_db:.4f}")

#обьеденяем все вместе и высчитываем w и b

def gradient_decent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_inters):
    
    m = len(x)

    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range (num_inters):
        dj_dw, dj_db = gradient_function (x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i<100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i% math.ceil(num_inters/10) == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.5f}")
    
    return w, b, J_history, w_history

initial_w = np.array([0,0])
initial_b = 0

iterations = 10000
alpha = 0.001

w,b,_,_ = gradient_decent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations) #_ это неиспользуемая переменная, так как функция возвращает нам еще и историю
print("w, b found by gradient decent: ", w, b)

#строим график на основе данных

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range (m):
    predicted[i] = np.dot(w, X_train[i]) + b

x_plot = np.linspace(x_train.min(), x_train.max(), 100)

X_plot = np.array([[x, x**2] for x in x_plot])
y_plot = X_plot @ w + b

plt.plot(x_plot, y_plot, c="b")
plt.scatter(x_train, y_train, marker="x", c="r")
plt.title("Moisure and Rain")
plt.ylabel("Rain?")
plt.xlabel("Pesent of moisure")
plt.show()