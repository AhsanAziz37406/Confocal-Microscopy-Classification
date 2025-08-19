function route = jRouletteWheelSelection(prob)
    % Handle cases where prob is empty or all elements are zero
    if isempty(prob) || all(prob == 0)
        route = 1;  % Assign a default value
        return;
    end

    % Original implementation
    c = cumsum(prob);
    p = rand() * c(end);
    route = find(c >= p, 1, 'first');
end
