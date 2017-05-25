corners = [0 0; 0 3; 1 3; 1 2; 2 2; 2 3; 3 3; 3 0; 2 0; 2 1; 1 1; 1 0];
number_of_corners = size(corners, 1);

number_of_neurons = number_of_corners;
neurons = rand(number_of_neurons, 2);

EPOCHS = 50000;
LEARNING_RATE = 0.1;
NEIGHBOURHOOD_RADIUS = number_of_neurons / 2;


current_learning_rate = LEARNING_RATE;
current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS;
    

for iteration = 1:EPOCHS
    % Plot the cities and neurons
    plot(corners(:, 1), corners(:, 2), '-b.', neurons(:, 1), neurons(:, 2), '-r.', 'markersize', 16);
    hold on;
    plot([corners(number_of_corners, 1), corners(1, 1)], [corners(number_of_corners, 2), corners(1, 2)], '-b.');
    plot([neurons(number_of_neurons, 1), neurons(1, 1)], [neurons(number_of_neurons, 2), neurons(1, 2)], '-r.');
    hold off;
    axis([0 3.5 0 3.5])
    pause(.001)
    title('SOM TSP')
    
    % Choose random input
    [current_corner, id] = datasample(corners, 1);

    % Calculate nearest city
    [winner_neuron, index] = find_best_matching_unit(neurons, current_corner);
       
    % Update weights of current neuron (and neighboorhood)
    for j = round(index-current_neighbourhood_radius):round(index+current_neighbourhood_radius)
        if j < 1 || j > number_of_corners
            continue
        end
        
        distance_to_winner_neuron = (j - index) ^ 2;

        % Neighbourhood Function
        sigma = current_neighbourhood_radius ^ 2;
        neighbouring_function_value = exp(-(distance_to_winner_neuron / sigma));

        % Update weights
        for i = 1:size(current_corner, 2)
            distance_to_corner = current_corner(i) - neurons(j, i);
            delta_weight = current_learning_rate * neighbouring_function_value * (distance_to_corner);

            neurons(j, i) = neurons(j, i) + delta_weight;
        end
    end

    % Update learning parameters
    current_learning_rate = LEARNING_RATE * (1 - iteration / EPOCHS);
    current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS * (1 - iteration / EPOCHS);
end
