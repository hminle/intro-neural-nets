corners = [0 0; 0 3; 1 3; 1 2; 2 2; 2 3; 3 3; 3 0; 2 0; 2 1; 1 1; 1 0];
number_of_corners = size(corners, 1);

data_points = rand(100, 2);
data_points = [data_points; rand(100, 2) + [0 1]];
data_points = [data_points; rand(100, 2) + [0 2]];
data_points = [data_points; rand(100, 2) + [1 1]];
data_points = [data_points; rand(100, 2) + [2 0]];
data_points = [data_points; rand(100, 2) + [2 1]];
data_points = [data_points; rand(100, 2) + [2 2]];
number_of_data_points = size(data_points, 1);

number_of_neurons = 30;
neurons = rand(number_of_neurons, 2) + [1 1];

EPOCHS = number_of_data_points * 10;
LEARNING_RATE = 0.1;
NEIGHBOURHOOD_RADIUS = number_of_neurons / 2;


current_learning_rate = LEARNING_RATE;
current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS;
    

for iteration = 1:EPOCHS
    if iteration == 0 || mod(iteration, 10) == 0
        disp(['Iteration: ', num2str(iteration)]);
        
        % Plot the corners, data points and neurons
        plot(corners(:, 1), corners(:, 2), '-b.', data_points(:, 1), data_points(:, 2), 'b.', neurons(:, 1), neurons(:, 2), '-r.', 'markersize', 16);
        hold on;
        plot([corners(number_of_corners, 1), corners(1, 1)], [corners(number_of_corners, 2), corners(1, 2)], '-b.');
        % plot([neurons(number_of_neurons, 1), neurons(1, 1)], [neurons(number_of_neurons, 2), neurons(1, 2)], '-r.');
        hold off;
        axis([0 3.5 0 3.5])
        pause(.001)
        title('SOM TSP')
    end
        
    % Choose random input
    [current_point, id] = datasample(data_points, 1);

    % Calculate nearest city
    [winner_neuron, index] = find_best_matching_unit(neurons, current_point);

    % Update weights of current neuron (and neighboorhood)
    for j = round(index-current_neighbourhood_radius):round(index+current_neighbourhood_radius)
        if j < 1 || j > number_of_neurons
            continue
        end

        distance_to_winner_neuron = (j - index) ^ 2;

        % Neighbourhood Function
        sigma = current_neighbourhood_radius ^ 2;
        neighbouring_function_value = exp(-(distance_to_winner_neuron / sigma));

        % Update weights
        for i = 1:size(current_point, 2)
            distance_to_point = current_point(i) - neurons(j, i);
            delta_weight = current_learning_rate * neighbouring_function_value * (distance_to_point);

            neurons(j, i) = neurons(j, i) + delta_weight;
        end
    end

    % Update learning parameters
    current_learning_rate = LEARNING_RATE * (1 - iteration / EPOCHS);
    current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS * (1 - iteration / EPOCHS);  
end

disp('Finito!')
