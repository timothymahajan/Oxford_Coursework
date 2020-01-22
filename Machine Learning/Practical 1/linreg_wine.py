import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error

X, y = cp.load(open('winequality-white.pickle', 'rb'))
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

# HANDIN 1
def plotYDistribution():
    # get unique data and count of each unique data and construct frequency array with tuples
    unique, count = np.unique(y_train, return_counts=True)
    frequency = np.array(list(zip(unique, count)))

    # determine x and y value of each tuple, and convert to list
    x_axis = np.arange(1, len(frequency)+1)
    x_freq = [x for (x, y) in frequency]
    y_freq = [y for (x, y) in frequency]

    # plot graph
    width = 0.5
    bar1 = plt.bar(x_axis, y_freq, width, color="r")
    plt.xlabel("Wine Quality")
    plt.xticks(x_axis + width/20.0, x_freq)
    plt.title("Distribution of White Wine Quality")
    plt.ylabel("Quantity")
    plt.show()

plotYDistribution()

'''
MSE of ZeroR on training set: 0.7768
MSE of ZeroR on training set: 0.8139
'''

# Handin 2

# calculate MSE according to formula 
def mse(y_set):
    mean = np.mean(y_train)
    sumMSE = 0
    for y in y_set:
        sumMSE += (y-mean)**2
    return sumMSE / len(y_set)


print('MSE of ZeroR on training set: %.4f' % mse(y_train))
print('MSE of ZeroR on training set: %.4f' % mse(y_test))


# HANDIN 3

'''
MSE of LinReg on training set: 0.5640
MSE of LinReg on test set: 0.5607
'''


# Scale training set to have mean=0 and std=1, scale testset with same values
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

# Train linear classifier
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)

# Get MSE of LinReg
mse_linreg_train = mean_squared_error(y_train, y_pred_train)
mse_linreg_test = mean_squared_error(y_test, y_pred_test)

print('MSE of LinReg on trainig set: %.4f' % mse_linreg_train)
print('MSE of LinReg on test set: %.4f' % mse_linreg_test)

# HANDIN 4

training_sizes = range(20, 600, 20)
MSE_training_list = []
MSE_test_list = []
for training_size in range(20, 600, 20):
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train[:training_size], y_train[:training_size])
    y_pred_train = lin_reg.predict(X_train[:training_size])
    y_pred_test = lin_reg.predict(X_test)
    MSE_train = mean_squared_error(y_train[:training_size], y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    MSE_training_list.append(MSE_train)
    MSE_test_list.append(MSE_test)


# Plot learning curve

plt.plot(training_sizes, MSE_training_list,'--', label="Training error")
plt.plot(training_sizes, MSE_test_list, label="Test error")
plt.title("Learning Curve for LinearRegression")
plt.xlabel("Training Set Size"), plt.ylabel("Mean Squared Error"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
