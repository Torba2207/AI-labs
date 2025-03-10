import numpy as np
import matplotlib.pyplot as plt
import random

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
x_new=np.c_[np.ones((len(x_train),1)),x_train]
x_new_transposed=np.transpose(x_new)
#theta_best=np.matmul(np.matmul(np.linalg.inv(np.matmul(x_new,x_new_transposed)),x_new_transposed),y_train)
inverted_x_xt=np.linalg.inv(x_new_transposed@x_new)
x_mul_inv=inverted_x_xt@x_new_transposed
theta_best=x_mul_inv@y_train
print(theta_best[0],theta_best[1])
#theta_best = [0, 0]

# TODO: calculate error
#mse_train=(1/len(x_train))*np.sum((theta_best[1]*x_train+theta_best[0]-y_train)**2)
#print(mse_train)

mse_test=(1/len(x_test))*np.sum((theta_best[1]*x_test+theta_best[0]-y_test)**2)
print(mse_test)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
#print(x_train)
std_sigma=np.std(x_train)
avg_miu=np.mean(x_train)
x_normalized=(x_train-avg_miu)/std_sigma
#print(x_normalized)

std_sigma_y=np.std(y_train)
avg_miu_y=np.mean(y_train)
y_normalized=(y_train-avg_miu_y)/std_sigma_y
#print(y_normalized)


# TODO: calculate theta using Batch Gradient Descent
learning_rate = 0.01
num_iterations = 1000
theta_best=[0,0]
def gradientMSE (theta,x,y):
    m=len(x)
    x_new = np.c_[np.ones((len(x), 1)), x]
    x_new_transposed = np.transpose(x_new)
    mse=(1/len(x))*np.sum((theta[1]*x+theta[0]-y)**2)
    print(mse)
    return ((2/m)*np.matmul(x_new_transposed,((np.matmul(x_new,theta))-y)))


gradientTheta=np.array([random.random(),random.random()])
m=len(x_normalized)
for _ in range (num_iterations):
    gradientTheta=gradientTheta-learning_rate*gradientMSE(gradientTheta,x_normalized,y_normalized)

#print(std_sigma)
#print(std_sigma_y)

#print(avg_miu)
#print(avg_miu_y)

theta_best[1]=gradientTheta[1]
theta_best[0]=gradientTheta[0]




# TODO: calculate error

#print(x_train)
std_sigma=np.std(x_train)
avg_miu=np.mean(x_train)
x_test_std=(x_test-avg_miu)/std_sigma
#print(x_normalized)

std_sigma_y=np.std(y_train)
avg_miu_y=np.mean(y_train)

y_test_std=(y_test-avg_miu_y)/std_sigma_y

#print(y_normalized)

#theta_best[1]=gradientTheta[1]*std_sigma_y/std_sigma
#theta_best[0]=avg_miu_y-theta_best[1]*avg_miu

y_pred_st=theta_best[0]+x_test_std*theta_best[1]

y_pred=y_pred_st*std_sigma_y+avg_miu_y

#mse_train=(1/len(x_train))*np.sum((theta_best[1]*x_train+theta_best[0]-y_train)**2)
#print(mse_train)

mse_test=(1/len(x_test))*np.sum((y_pred-y_test)**2)
print(mse_test)

# plot the regression line
x = np.linspace(min(x_test_std), max(x_test_std), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test_std, y_test_std)
plt.xlabel('Weight scaled')
plt.ylabel('MPG scaled')
plt.show()