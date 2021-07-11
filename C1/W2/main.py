import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    listClass = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, listClass


# load data
x_train, y_train, x_test, y_test, classes = load_dataset()
print(x_train.shape)  # (209, 64, 64, 3) -- 209 images 64x64
print(y_train.shape)  # (1, 209)
print(x_test.shape)  # (50, 64, 64, 3) -- 50 images 64x64
print(y_test.shape)  # (1, 50)
print(classes)  # [b'non-cat' b'cat']

# test print a image
# test_index = 49
# plt.imshow(x_train[test_index])
# print("this is a " + classes[y_train[0][test_index]].decode("utf-8"))
# plt.show()

# reshape
x_train = x_train.reshape(x_train.shape[0], -1).T
x_test = x_test.reshape(x_test.shape[0], -1).T
print("train_set_x_flatten shape: " + str(x_train.shape))
print("train_set_y shape: " + str(x_test.shape))

# standardized
x_train = x_train / 255
x_test = x_test / 255


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def propagate(w, b, X, Y):
    m = X.shape[1]  # number example : 209
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation, w.T : 1x12288, X: 12288x209
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = 0
    db = 0
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):  # notice: function predict all of image in set, not one image, Y_prediction is a set!!!
    m = X.shape[1]  # number example : 209
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w = np.zeros(shape=(X_train.shape[0], 1))  # X_train.shape[0] = 12288
    b = 0
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: " + str(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: " + str(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    return {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}


myModel = model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.005, print_cost=True)
# done

# test recognize cat in test set
index_test = 12
Y_prediction_test = myModel["Y_prediction_test"]
isCat = int(Y_prediction_test[0][index_test])  # 0 or 1
print("this is a " + classes[isCat].decode("utf-8"))
plt.imshow(x_test[:, index_test].reshape((64, 64, 3)))
plt.show()

# Plot learning curve (with costs)
costs = np.squeeze(myModel['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(myModel["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(x_train, y_train, x_test, y_test, num_iterations=1500, learning_rate=i, print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
