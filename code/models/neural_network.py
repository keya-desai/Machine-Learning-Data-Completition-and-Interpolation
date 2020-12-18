import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNetwork:
    def __init__(self, x, y, valX, valY, featureVal,  num_hidden, epochs, learning_rate, num_nodes_layers, activation_function, batch_size):
        self.x = x
        self.y = y
        
        self.valX = valX
        self.valY = valY
        self.featureVal = featureVal

        self.num_data = np.shape(x)[1]  # no. of data points    # no. of rows
        self.k = np.shape(x)[0]  # no. of features   # no. of cols
        self.n_out = np.shape(y)[0]

        self.batch_size = batch_size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_hidden = num_hidden
        self.num_layers = num_hidden + 1  # +1 for output layer

        self.num_nodes_layers = num_nodes_layers

        # inserting input and output nodes to the list
        self.num_nodes_layers.insert(0, self.k)
        self.num_nodes_layers.append(self.n_out)

        self.leaky_slope = 0.01
        self.weights = []
        
        # parameters: weight and bias
        # weight[l] : (num_layers * num_layers-1 ) * num_layers : (no. of nodes in layer l * no. of nodes in layer (l-1)) * no. of layers
    def initialize_parameters_random(self):

        for l in range(1, self.num_layers + 1):
            self.weights.append(
                np.random.rand(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]))

    # Use this when activation function is tanh or sigmoid
    def initialize_parameters_xavier(self):

        for l in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]) * np.sqrt(
                1 / self.num_nodes_layers[l - 1]))

    # Use this when activation function is ReLU or Leaky ReLu
    def initialize_parameters_he(self):
        for l in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(self.num_nodes_layers[l], self.num_nodes_layers[l - 1]) * np.sqrt(
                2 / self.num_nodes_layers[l - 1]))

    # Activation Functions
    def activation(self, x):
        if self.activation_function == "linear":
            return x
        if self.activation_function == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.activation_function == "tanh":
            return np.tanh(x)
        if self.activation_function == "relu":
            a = np.zeros_like(x)
            return np.maximum(a, x)
        if self.activation_function == "leaky_relu":
            a = self.leaky_slope * x
            return np.maximum(a, x)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def gradient_activation(self, X):
        if self.activation_function == "linear":
            return np.ones_like(X)
        elif self.activation_function == "sigmoid":
            return self.activation(X) * (1 - self.activation(X))
        elif self.activation_function == "tanh":
            return (1 - np.square(X))
        elif self.activation_function == "relu":
            grad = np.zeros_like(X)
            grad[X > 0] = 1.0
            return grad
        elif self.activation_function == "leaky_relu":
            grad = np.ones_like(X)
            grad[X <= 0] = self.leaky_slope
            return grad

    def forward_propogation(self, x):
        # dim of A vector: (no. of hidden nodes * num_data) *(no. of layers)
        A = []
        Z = []
        A.append(x)
        A_prev = x

        for l in range(0, self.num_layers-1):
            z = np.matmul(self.weights[l], A_prev)
            a = self.activation(z)
            A_prev = a
            A.append(a)
            Z.append(z)
        z = np.matmul(self.weights[-1], A_prev)
        # ******* Can apply different activation to differnt nodes in last layer?****
        a = self.activation(z)
        A.append(a)
        Z.append(z)
        return (A, Z)

    def back_propogation(self, A, Z, y):

        delta_z = [None for i in range(self.num_layers)]
        delta_weight = [None for i in range(self.num_layers)]

        delta_z[-1] = (y - A[-1])
        delta_weight[-1] = np.matmul(delta_z[-1], A[-2].T)

        for l in range(self.num_layers - 2, -1, -1):
            delta_z[l] = np.multiply(np.matmul(self.weights[l + 1].T, delta_z[l + 1]), self.gradient_activation(Z[l]) )
            delta_weight[l] = np.matmul( delta_z[l], A[l].T )

        return delta_weight


    def update_weight(self, A, delta_weight):
        # weight = weight + learning_rate * error * input
        m = A[-1].shape[1]
        for l in range(self.num_layers):
            self.weights[l] = self.weights[l] + (self.learning_rate * delta_weight[l])/m

    def predict(self, x_test, isMissing):
        A,Z = self.forward_propogation(x_test)
        prediction = A[-1]
        predFinal = np.where(isMissing < 1, prediction, x_test)
        return predFinal

    def loss_function(self, y, out):
#             return (0.5 * np.mean((y - out) ** 2))
        return (np.mean(np.sum((y - out) ** 2, axis = 1)))

    def model(self):
        mini_batch = int((self.num_data) / (self.batch_size))
        
        self.initialize_parameters_random()
        
#         if self.activation_function == "linear":
#             self.initialize_parameters_random()
#         elif self.activation_function == "sigmoid" or self.activation_function == "tanh":
#             self.initialize_parameters_xavier()
#         else:
#             self.initialize_parameters_he()
        leastVal = 1000
        numBadEpoch = 0
        trainLossArr = []
        testLossArr = []
        
        for e in range(self.epochs):
            trainLossBatch = []
            print("Epoch =", e)
            end = 0
            for n in range(mini_batch + 1):
                if (n != mini_batch):
                    start = n * self.batch_size
                    end = (n + 1) * self.batch_size
                    x_ = self.x[:, start:end]
                    y_ = self.y[:, start:end]

                else:
                    if ((self.num_data % self.batch_size) != 0):
                        x_ = self.x[:, end:]
                        y_ = self.y[:, end:]
                    else:
                        break

                A,Z = self.forward_propogation(x_)
                delta_weight = self.back_propogation(A, Z, y_)
                self.update_weight(A, delta_weight)
                trainLoss = self.loss_function(A[-1], y_)
                trainLossBatch.append(trainLoss)
                
            pred = self.predict(self.valX, self.featureVal)
            valLoss = self.loss_function(self.valY, pred)
            
            if valLoss < leastVal:
                leastVal = valLoss
                print("** Least Validation Loss")
                numBadEpoch = 0
            else:
                print("Bad Epoch!")
                numBadEpoch += 1
            if numBadEpoch == 5:
                break
            
            trainLossArr.append(sum(trainLossBatch)/len(trainLossBatch))
            testLossArr.append(valLoss)
            
            print("Train loss = ", trainLossArr[-1])
            print("Validation loss = ", valLoss)
        return trainLossArr, testLossArr