import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

# Calculates the total distance for a solution
def _distance(tsp_solution, distance_matrix):
    total_distance = 0
    for i in range(len(tsp_solution)-1):
        total_distance += (distance_matrix[tsp_solution[i]][
            tsp_solution[i + 1]])
    total_distance += (distance_matrix[tsp_solution[0]][
        tsp_solution[-1]])
    return total_distance


# Uses 3-opt algorithm to find a better solution in the neighboourhood
def _3_opt(route, distance_matrix):
    current_dist = _distance(route, distance_matrix)

    # Find three connections to break
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route) - 1):
            for k in range(j + 2, len(route) - 1):
                # Now reconnect the graph in every other way possible,
                # and see if any gives a better solution

                # Way 1
                temp = route[:]
                temp[j+1:k+1] = reversed(route[j+1:k+1])
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 2
                temp = route[:]
                temp[i+1:j+1] = reversed(route[i+1:j+1])
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 3
                temp = route[:]
                temp[i+1:j+1], temp[j+1:k+1] = (
                    reversed(route[i+1:j+1]), reversed(route[j+1:k+1]))
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 4
                temp = (route[:i+1] + route[j+1:k+1] +
                        route[i+1:j+1] + route[k+1:])
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 5
                temp = route[:i+1] + route[j+1:k+1]
                temp += reversed(route[i+1:j+1])
                temp += route[k+1:]
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 6
                temp = route[:i+1]
                temp += reversed(route[j+1:k+1])
                temp += route[i+1:j+1]
                temp += route[k+1:]
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist

                # Way 7
                temp = route[:i+1]
                temp += reversed(route[j+1:k+1])
                temp += reversed(route[i+1:j+1])
                temp += route[k+1:]
                dist = _distance(temp, distance_matrix)
                if dist < current_dist:
                    route = temp
                    current_dist = dist
    return route


# Uses a Circular Self-Organizing Map with a Gaussian Neighbourhood function and linearly
# decreasing learning rate, to solve TSP.
class TSPSolver(object):
    _trained = False

    # Initializes all necessary components of the TensorFlow Graph.
    def __init__(self, input_vects, n_iterations=200):
        self._n = len(input_vects)
        dim = len(input_vects[0])
        self._n_centroids = self._n*3
        alpha = 0.3
        sigma = 0.3
        self._n_iterations = abs(int(n_iterations))

        # Initialization of the centroid (neuron) vectors
        init_values = []
        for vect in input_vects:
            init_values.extend(list(vect))
        init_mean = np.mean(init_values)
        init_dev = np.sqrt(np.var(init_values))

        # Initialize Graph
        self._graph = tf.Graph()

        # Set up the graph with all the components
        with self._graph.as_default():
            #Randomly initialized weights for all neurons
            self._weightage_vects = tf.Variable(
                tf.random_normal([self._n_centroids, dim], init_mean, init_dev)
            )

            # SOM grid locations of neurons
            self._location_vects = tf.constant(
                np.array(list(self._neuron_locations(self._n_centroids)))
            )

            # Training vector
            self._vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self._iter_input = tf.placeholder("float")

            # Construct train_op piece by piece

            # Compute the Best Matching Unit (BMU) given a vector
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(
                        self._n_centroids)])), 2), 1)),
                                  0)

            # Extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))), [2])

            # Cmpute the alpha and sigma values based on number of iterations
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning rates for all neurons
            # based on iteration number and location with respect to BMU
            bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(self._location_vects,
                tf.stack([bmu_loc for i in range(self._n_centroids)])), 2), 1
            )
            neighbourhood_func = tf.exp(tf.negative(tf.div(
                tf.cast(bmu_distance_squares, "float32"),
                tf.pow(_sigma_op, 2)))
            )
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Update weights using the learning_rate_op
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(learning_rate_op, np.array([i]),
                np.array([1])), [dim]) for i in range(self._n_centroids)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(self._n_centroids)]),
                    self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects, weightage_delta)
            self._training_op = tf.assign(self._weightage_vects, new_weightages_op)

            # Initialize session and variables
            self._sess = tf.Session()
            self._sess.run(tf.initialize_all_variables())

            # Train the solver
            self.train(input_vects)

    # Returns the TSP solution order as a list and the total Euclidean distance
    @property
    def solution(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._sol, self._sol_dist

    # Returns the locations of the individual neurons in the SOM
    @classmethod
    def _neuron_locations(cls, n_centroids):
        for i in range(n_centroids):
            yield np.array([np.cos(i*2*np.pi/float(n_centroids)),
                            np.sin(i*2*np.pi/float(n_centroids))])

    # Trains the SOM
    def train(self, input_vects):
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect, self._iter_input: iter_no})
        self._input_vects = input_vects
        self._compute_solution()
        self._trained = True

    # Computes the solution to the TSP
    def _compute_solution(self):
        centroid_vects = list(self._sess.run(self._weightage_vects))
        centroid_locations = list(self._sess.run(self._location_vects))

        # Distance matrix mapping input point number to list of distances from centroids
        distances = {}
        for point in range(self._n):
            distances[point] = []
            for centroid in range(self._n_centroids):
                distances[point].append(
                    np.linalg.norm(centroid_vects[centroid] -
                                   self._input_vects[point]))

        # Distance matrix mapping input point number to list of distances from other input points
        point_distances = {}
        for point in range(self._n):
            point_distances[point] = []
            for other_point in range(self._n):
                point_distances[point].append(
                    np.linalg.norm(self._input_vects[other_point] -
                                   self._input_vects[point]))

        # Compute angle with respect to each city (point)
        point_angles = {}
        for point in range(self._n):
            total_vect = 0
            cents = [j for j in range(self._n_centroids)]
            cents.sort(key=lambda x: distances[point][x])
            for centroid in cents[:2]:
                total_vect += (1.0/(distances[point][centroid]) *
                               centroid_locations[centroid])
            total_vect = total_vect/np.linalg.norm(total_vect)
            if total_vect[0] > 0 and total_vect[1] > 0:
                angle = np.arcsin(total_vect[1])
            elif total_vect[0] < 0 and total_vect[1] > 0:
                angle = np.arccos(total_vect[0])
            elif total_vect[0] < 0 and total_vect[1] < 0:
                angle = np.pi - np.arcsin(total_vect[1])
            else:
                angle = 2*np.pi - np.arccos(total_vect[0])
            point_angles[point] = angle

        # Find the rough solution
        tsp_solution = [i for i in range(self._n)]
        tsp_solution.sort(key=lambda x: point_angles[x])
        tsp_solution = _3_opt(tsp_solution, point_distances)

        # Compute the total distance for the solution
        total_distance = _distance(tsp_solution, point_distances)

        self._sol = tsp_solution
        self._sol_dist = total_distance
