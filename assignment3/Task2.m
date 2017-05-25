cities = [0.2 0.1; 0.15 0.2; 0.4 0.45; 0.2 0.77; 0.5 0.9; 0.83 0.65;
            0.7 0.5; 0.82 0.35; 0.65 0.23; 0.6 0.28];
number_of_cities = size(cities, 1);
        
number_of_neurons = number_of_cities;
neurons = rand(number_of_neurons, 2);

EPOCHS = 50000;
LEARNING_RATE = 0.1;
NEIGHBOURHOOD_RADIUS = number_of_neurons / 2;


current_learning_rate = LEARNING_RATE;
current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS;


for iteration = 1:EPOCHS
    % Plot the cities and neurons
    plot(cities(:,1), cities(:,2), 'r.', neurons(:,1), neurons(:,2), '-b.', 'markersize', 16);
    hold on;
    plot([neurons(number_of_neurons, 1), neurons(1, 1)], [neurons(number_of_neurons, 2), neurons(1, 2)], '-b.', 'markersize', 16);
    hold off;
    axis([0 1 0 1])
    pause(.001)
    title('SOM TSP')
    
    % Choose random input
    [current_city, id] = datasample(cities, 1);

    % Calculate nearest city
    [winner_neuron, index] = find_best_matching_unit(neurons, current_city);

    % Update weights of current neuron (and neighboorhood)
    for j = round(index-current_neighbourhood_radius):round(index+current_neighbourhood_radius)
        if j < 1 || j > number_of_cities
            continue
        end
        
        distance_to_winner_neuron = (j - index) ^ 2;

        % Neighbourhood Function
        sigma = current_neighbourhood_radius ^ 2;
        neighbouring_function_value = exp(-(distance_to_winner_neuron / sigma));

        % Update weights
        for i = 1:size(current_city, 2)
            distance_to_city = current_city(i) - neurons(j, i);
            delta_weight = current_learning_rate * neighbouring_function_value * (distance_to_city);

            neurons(j, i) = neurons(j, i) + delta_weight;
        end
    end

    % Update learning parameters
    current_learning_rate = LEARNING_RATE * (1 - iteration / EPOCHS);
    current_neighbourhood_radius = NEIGHBOURHOOD_RADIUS * (1 - iteration / EPOCHS);
end
