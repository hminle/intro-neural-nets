# Package imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from argparse import ArgumentParser

# Import Model
from two_layers_mlp import TwoLayersNeuralNet

# Default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

args_parser = ArgumentParser(description='specify N and T')
args_parser.add_argument('--n', default=1, type=int, metavar='<N>', 
        help='Define N, default=1')
args_parser.add_argument('--t', default=1, type=int, metavar='<T>',
        help='Define T, default=1')
args_parser.add_argument('--train_percent', default=1, type=float, 
        metavar='<train_percentage>', help='Define train percentage, default=1')
args_parser.add_argument('--validation_percent', default=0,type=float,
        metavar='<validation_percent>', 
        help='Define valid percentage, default=0')
args_parser.add_argument('--test_percent', default=0, type=float,
        metavar='<test percent>', help='Define test percent, default=0')
args_parser.add_argument('--epochs', default=50,type=int,
        metavar='<Number of iterations>', help='Define num of epochs, default=50')
args_parser.add_argument('--jump_connection', default=False, type=bool,
        metavar='<jump_connection_flag>', help='Define jump connection or not, default=False')
args_parser.add_argument('--random_data', default=False, type=bool,
        metavar='<random_data>', help='Define random data or not, default=False')
args = args_parser.parse_args()
## Prepare data
N = args.n
T = args.t
train_percent = args.train_percent
validation_percent = args.validation_percent
test_percent = args.test_percent
epochs = args.epochs
jump_connection = args.jump_connection
is_random = args.random_data


data = pd.read_csv("stock_price.csv", names=["Date", "Price"])
prices = np.vectorize(lambda price: re.sub(',', '', price))(data.Price.values)
prices = prices.reshape(prices.shape[0], 1)
prices = prices.astype('float32')

## Shuffle Data
if is_random:
   np.random.shuffle(prices)

# prepare the dataset of input to output pairs encoded as integers

def read_and_prepare_data(train_percent, validation_percent, test_percent, data):
    train_records = round(train_percent * len(data))
    validation_records = round(validation_percent * len(data))
    test_records = round(test_percent * len(data))
    # Divide the whole data into the three parts (training, validation, test)
    mask = range(train_records)
    train_data = data[mask]
    
    if validation_records > 0:
        mask = range(train_records, train_records + validation_records)
        validation_data = data[mask]
    else:
        validation_data = []
    
    if test_records > 0:
        test_data = data[(train_records + validation_records):]
    else:
        test_data = []

    return train_data, validation_data, test_data


def create_train_target(N, T, data):
    dataX = []
    dataY = []
    for i in range(0, len(data) - N - T, 1):
        seq_in = data[i:i + N]
        seq_out = data[i + N + T - 1]
        dataX.append([price for price in seq_in])
        dataY.append(seq_out)
    assert len(dataX) == len(dataY)
    return dataX, dataY


train_data, validation_data, test_data = read_and_prepare_data(train_percent, validation_percent, test_percent, prices)


X_train, y_train = create_train_target(N, T, train_data)
X_val, y_val = create_train_target(N, T, validation_data)
X_test, y_test = create_train_target(N, T, test_data)


X_train = np.reshape(X_train, (len(X_train), N))
y_train = np.reshape(y_train, (len(y_train), 1))
X_val = np.reshape(X_val, (len(X_val), N))
y_val = np.reshape(y_val, (len(y_val), 1))
X_test = np.reshape(X_test, (len(X_test), N))
y_test = np.reshape(y_test, (len(y_test), 1))

## Build Model

model = TwoLayersNeuralNet(N, 1, 5, jump_connection=jump_connection)

model.train(X_train ,y_train, X_val, y_val, X_test, y_test, 50, learning_rate=0.000000001)

print("Time to reach best performance in seconds " + str(model.time_to_reach_best_performance))
print("Best Loss " + str(model.best_performance))

# Show RMSE
figure1 = plt.figure(1)
plt.plot(model.train_loss, 'b-', label='Train Loss')
plt.plot(model.valid_loss, 'g-', label='Valid Loss')
plt.plot(model.test_loss, 'r-', label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss history')
plt.legend()
figure1.show()

# Show Pred data vs real data in test set
if len(X_test) > 0: 
    figure2 = plt.figure(2)
    plt.plot(model.predict(X_test), 'b-', label="Predicted Data")
    plt.plot(y_test, 'r-', label="Real Data")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Predicted Data vs Real Data in Test Data')
    plt.legend()
    figure2.show()

# Show Pred data vs real data in train set
figure3 = plt.figure(3)
plt.plot(model.predict(X_train), 'b-', label="Predicted Data")
plt.plot(y_train, 'r-', label="Real Data")
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted Data vs Real Data in Train Data')
plt.legend()
figure3.show()
## Show Extrapolate Data
#y_con = model.predict(X_train[-1])
#for i in range(249,277):
#    y_pred = model.predict(np.reshape(y_con[-1], (1,)))
#    y_con = np.concatenate((y_con, y_pred ), axis=0)
#

#y_con = np.concatenate((y_train, y_con), axis=0)
#figure4 = plt.figure(4)
#plt.plot(model.predict(X_train), 'b-', label="Predicted Data")
#plt.plot(y_con, 'g-', label="Extrapolate Data")
#plt.plot(y_train, 'r-', label="Real Data")
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.title('Extrapolate Data with Train Set')
#plt.legend()
#figure4.show()
#

input()
