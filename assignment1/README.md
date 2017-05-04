How to Execute
==============

Install the requirements with `pip3 install -r requirements.txt`.

### Task 1: Stock Prices

Start the program with `python3 task1.py`.
The program offers different arguments to achieve the different tasks needed for the assigment. They are all explained below.

1. `--n` to define the N
2. `--t` to define T
3. `--train_percent` to define the percentage of records that will be used for training
4. `--validation_percent` to define the percentage of records that will be used for validation
5. `--test_percent` to define the percentage of records that will be used for testing
6. `--epochs` to define the number of iterations
7. `--jump_connection` to set whether to use jumping connections or not
8. `--random_data` to define whether to use random data samples for training, validation and test.

Use `python3 task1.py --help` to see all the flags.


Example usage: `python3 task1.py --n 1 --t 1 --epochs 25 --jump_connection True --random_data True --train_percent 0.6 --test_percent 0.3 --validation_percent 0.1`


### Task 2: Two Moons

Start the program with `python3 two_moons.py`.

### Task 3: CIFAR10

Make sure that the six `data_batch_(#i)` as well as the `test_batch` are in the `images` folder.
After that start the programm with `python3 cifar10.py`.
