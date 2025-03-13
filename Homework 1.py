import pandas as pd
import numpy as np
import math
import copy
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
'''print(os.getcwd())
 '''

# Loading in Data
df_x_train = pd.read_excel('Problem Set 1/PS1_data.xlsx', sheet_name='X_train')
df_y_train = pd.read_excel('Problem Set 1/PS1_data.xlsx', sheet_name='Y_train')

# DataFrame to Numpy`
x_train = df_x_train.values.flatten()
y_train = df_y_train.values.flatten()

# Inspect Data
print(f'Type of Data X Value: {x_train.dtype}')
print("First five elements of x_train are:\n", x_train[:5])
print(f'Type of Data X Value: {y_train.dtype}')
print("First five elements of y_train are:\n", y_train[:5])
print('The shape of x_train is:', x_train.shape)
print('The shape of y_train is: ', y_train.shape)
print('Number of training examples (m):', len(x_train))
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# Exercise 1: Compute Cost
def compute_cost(x, y, w, b):
    total_cost = 0
    m = x_train.shape[0]
    for i in range(m):
        pred = (w * x[i]) + b
        cost = (pred - y[i]) ** 2
        total_cost += cost
    return total_cost/(2*m)

initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f'Cost at initial w: {cost:.3f}')

# Exercise 2: Gradient Descent
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        pred = (w * x[i]) + b
        dj_dw = dj_dw + (pred - y[i]) * x[i]
        dj_db = dj_db + (pred - y[i])
    return dj_dw/m , dj_db/m

test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)
print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    j_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        if i < 100000:
            cost = cost_function(x, y, w, b)
            j_history.append(cost)
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {j_history[-1]:8.2f} ")
    return w, b, j_history, w_history

initial_w = 0
initial_b = 0
iterations = 1500
alpha = 0.01
w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)
# Exercise 3 Plotting Linear Fit
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = (w * x_train[i]) + b
print("Predicted values:", predicted)
plt.scatter(x_train, predicted, marker='x', c='r')
plt.plot(x_train, predicted, c='b')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# Exercise 4 Testing Predictions
predict = []
for x_input in [35000, 70000]:
    prediction = (w * (x_input/10000)) + b
    predict.append(prediction)

print(f'For population = 35,000, we predict a profit of ${predict[0] * 10000:.2f}')
print(f'For population = 70,000, we predict a profit of ${predict[1] * 10000:.2f}')