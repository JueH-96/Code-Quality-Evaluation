import sys
from itertools import permutations

# Read the number of islands and bridges
N, M = map(int, sys.stdin.readline().split())

# Read the bridge information
bridges = [None] * M
for i in range(M):
    U, V, T = map(int, sys.stdin.readline().split())
    bridges[i] = (U, V, T)

# Read the number of queries
Q = int(sys.stdin.readline())

# Process each query
for _ in range(Q):
    # Read the number of bridges in the query and the bridges themselves
    K = int(sys.stdin.readline())
    query_bridges = list(map(int, sys.stdin.readline().split()))
    
    # Extract the relevant bridges for the query
    relevant_bridges = [bridges[b - 1] for b in query_bridges]
    
    # Initialize the minimum time to a large number
    min_time = float('inf')
    
    # Check all permutations of the relevant bridges to find the minimum time
    for perm in permutations(relevant_bridges):
        time = 0
        current_island = 1
        for bridge in perm:
            U, V, T = bridge
            # Check if the current bridge is connected to the current island
            if current_island == U:
                current_island = V
            elif current_island == V:
                current_island = U
            else:
                # If not connected, add the time to the nearest end of the bridge
                nearest_end_time = min(abs(current_island - U), abs(current_island - V))
                time += nearest_end_time
                current_island = V if nearest_end_time == abs(current_island - U) else U
            # Add the time to cross the bridge
            time += T
        # Check if we ended up at island N, if not add the time to get there
        if current_island != N:
            time += abs(current_island - N)
        # Update the minimum time if the current permutation is better
        min_time = min(min_time, time)
    
    # Print the minimum time for the current query
    print(min_time)