import numpy as np

from solver import TSPSolver

# (0.2, 0.1)   (0.15, 0.2)  (0.4, 0.45)   (0.2, 0.77)   (0.5, 0.9)
#     (0. 83, 0.65) (0.7, 0.5)   (0.82, 0.35)  (0.65, 0.23)  (0.6, 0.28).

input_vectors = np.array([[0.2, 0.1], [0.15, 0.2], [0.4, 0.45], [0.2, 0.77], [0.5, 0.9],
                          [0.83, 0.65], [0.7, 0.5], [0.7, 0.5], [0.82, 0.35], [0.65, 0.23],
                          [0.65, 0.23], [0.6, 0.28]])


print('Initial')
solver = TSPSolver(input_vectors, 1)
solution = solver.solution
print(solution)


print('')
print('Fair')
solver = TSPSolver(input_vectors, 5)
solution = solver.solution
print(solution)

print('')
print('Good')
solver = TSPSolver(input_vectors, 10)
solution = solver.solution
print(solution)

print('')
print('Best')
solver = TSPSolver(input_vectors, 200)
solution = solver.solution
print(solution)
