import numpy as np
from sklearn.metrics import mean_squared_error

class TwoLayersNeuralNets():
    def __init__(self, input_dim, output_dim, h_dim, seed_num=7):

        # Initialize the parameters
        np.random.seed(seed_num)
        self.W1 = np.random.randn(input_dim, h_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, h_dim))
        self.W2 = np.random.rand(h_dim, output_dim) / np.sqrt(h_dim)
        self.b2 = np.zeros((1, output_dim))
        self.loss = []

    # This function learns parameters for the neural network and returns the model.
    # - X: input data
    # - y: output target
    # - h_dim: Number of nodes in the hidden layer
    # - epochs: Number of passes through the training data for gradient descent
    def train(self, X, y, epochs, epsilon=0.000000001, reg_lambda=0.01):
        mean = (np.amax(y) - np.amin(y))/2 
        self.min_value = np.amin(y)
        self.max_value = np.amax(y)
        normFunction = np.vectorize(self.normalize)

        # Gradient descent. For each batch...
        for i in range(0, epochs):
    
            # Forward
            z1 = X.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            a1_norm = normFunction(a1, self.min_value, self.max_value)
            z2 = a1_norm.dot(self.W2) + self.b2
            
            output = np.around(z2)

            # Backpropagation
            delta3 = output - y

            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))*mean
            #delta2 = delta3.dot(self.W2.T)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * self.W2
            dW1 += reg_lambda * self.W1
    
            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon *db1
            self.W2 += -epsilon *dW2
            self.b2 += -epsilon *db2
            loss = self.calculate_loss(output, y, reg_lambda)
            self.loss.append(loss)
            print("RMSError after iteration %i: %f" %(i, loss))
    
    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self, X, y, reg_lambda):
        output = X
        num_examples = output.shape[0]
        output = np.nan_to_num(output)
        y = np.nan_to_num(y)
        
        data_loss = np.sqrt(1./num_examples * mean_squared_error(output, y))
        return np.log(data_loss)

       
    def normalize(self, x, min_value, max_value):
        mean = (max_value - min_value)/2.
        if x >= -1 and x <= 0:
            return -1*((x + 1)*mean - min_value )
        else:
            return (x - 1)*mean + max_value

    def forward(self, X):
         normFunction = np.vectorize(self.normalize)

         z1 = X.dot(self.W1) + self.b1
         a1 = np.tanh(z1)
         a1_norm = normFunction(a1, self.min_value, self.max_value)
         z2 = a1_norm.dot(self.W2) + self.b2
         return np.round(z2)


    # Helper function to predict an output (0 or 1)
    def predict(self, x):
        return self.forward(x) 
