def solve_n_M_at_most_x1(train_departs_city, train_reaches_city, train_departs_time, train_reaches_time, M, maximum_delay):
    """
    Solves the problem for N <= M (which is always true in given constraints) and delay <= maximum_delay.
    """
    # Set of nodes with no incoming edges.
    start_nodes = set(train_departs_city) - set(train_reaches_city)
    
    # Set up for level computation.
    levels = [-1] * (M + 1)
    for i in start_nodes:
        levels[i] = 0
    
    # Set up for calculating shortest paths.
    min_costs = [0] * (M + 1)
    
    result = []
    
    while start_nodes:
        current_node = start_nodes.pop()
        min_age = min_costs[current_node]
        for next_node, age_difference in zip(
            train_reaches_city, (train_reaches_time - train_departs_time)
        ):
            if next_node == -1:
                result.append(max(0, (min_age - current_node) - age_difference))
            else:
                if min_costs[next_node] > min_age - age_difference + maximum_delay:
                    min_costs[next_node] = min_age - age_difference + maximum_delay
                    if levels[next_node] == -1:
                        levels[next_node] = levels[current_node] + 1
                        start_nodes.add(next_node)
    
    return result

# Reading inputs
N, M, X_1 = map(int, input().split())
train_data = [list(map(int, input().split())) for _ in range(M)]
train_departs_city, train_reaches_city, train_departs_time, train_reaches_time = zip(*train_data)

# Convert to 0-indexed for python.
train_departs_city = [x - 1 for x in train_departs_city]
train_reaches_city = [x - 1 if x != -1 else -1 for x in train_reaches_city]

result = solve_n_M_at_most_x1(train_departs_city, train_reaches_city, train_departs_time, train_reaches_time, M, X_1)
print(' '.join(map(str, result)))