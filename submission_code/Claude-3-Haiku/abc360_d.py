from collections import defaultdict

def count_passing_pairs(N, T, S, X):
    # Create a dictionary to store the positions of each ant at each time step
    positions = defaultdict(list)
    
    # Initialize the positions of the ants at time 0
    for i in range(N):
        positions[0].append((X[i], i, int(S[i])))
    
    # Simulate the movement of the ants
    for t in range(1, int(T + 0.1) + 1):
        for i in range(N):
            x, ant_id, direction = positions[t-1][i]
            new_x = x + (1 if direction else -1)
            positions[t].append((new_x, ant_id, direction))
    
    # Count the number of passing pairs
    count = 0
    for t in range(1, int(T + 0.1) + 1):
        positions_at_t = positions[t]
        for i in range(N):
            for j in range(i+1, N):
                x1, ant1, dir1 = positions_at_t[i]
                x2, ant2, dir2 = positions_at_t[j]
                if (x1 < x2 and dir1 and not dir2) or (x1 > x2 and not dir1 and dir2):
                    count += 1
    
    return count