function distance = calculate_distance(city, neuron)
    distance = 0;
    for i = 1:size(city, 2)
        distance = distance + (city(i) - neuron(i)) ^ 2;
    end
    distance = sqrt(distance);
end
