function [winner, index] = find_best_matching_unit(neurons, city)
    lowest_distance = 9999;
    
    for i = 1:size(neurons, 1)
        neuron = neurons(i,:);
        distance = calculate_distance(city, neuron);
        if distance < lowest_distance
            lowest_distance = distance;
            winner = neuron;
            index = i;
        end
    end
end
