import imageio
import matplotlib.pyplot as plt


from my_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
x_train, y_train, x_test, y_test, classes = load_data()
print(x_train.shape)  # (209, 64, 64, 3)
print(y_train.shape)  # (1, 209)
print(x_test.shape)  # (50, 64, 64, 3)
print(y_test.shape)  # (1, 50)

# # test train set
# index = 10
# plt.imshow(x_train[index])
# print("this is a ", classes[y_train[0][index]].decode("utf-8"))
# plt.show()

m_train = x_train.shape[0]  # 209
num_px = x_train.shape[1]  # 64
m_test = x_test.shape[0]  # 50

# Reshape the training and test examples
train_x_flatten = x_train.reshape(x_train.shape[0], -1).T  # -1 makes reshape flatten the remaining dimensions
test_x_flatten = x_test.reshape(x_test.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
x_train = train_x_flatten / 255.
x_test = test_x_flatten / 255.

print(str(x_train.shape))  # (12288, 209)
print(str(x_test.shape))  # (12288, 50)

n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# parameters = two_layer_model(x_train, y_train, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True)
#
# predictions_train = predict(x_train, y_train, parameters)
#
# predictions_test = predict(x_test, y_test, parameters)


# --------------------L layer----------------------

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

layers_dims = [12288, 20, 7, 5, 1]

parameters = L_layer_model(x_train, y_train, layers_dims, num_iterations=2500, print_cost=True)

pred_train = predict(x_train, y_train, parameters)

pred_test = predict(x_test, y_test, parameters)

# my_image = "1.jpg"
# my_label_y = [1]
#
# url = "my_test_cats/" + my_image
# image = imageio.imread(url)
# my_image = image.reshape((num_px * num_px * 3, 1))
# my_image = my_image / 255.
# my_predicted_image = predict(my_image, my_label_y, parameters)
#
# plt.imshow(image)
# print("this is a " + classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8"))
