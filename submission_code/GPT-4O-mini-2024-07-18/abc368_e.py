def find_delays(N, M, X1, trains):
    from collections import defaultdict
    import heapq

    # Create a graph where edges represent the transfer conditions
    graph = defaultdict(list)
    
    for i in range(M):
        A, B, S, T = trains[i]
        graph[B].append((A, S, T, i))  # B_i -> (A_i, S_i, T_i, index)

    # Initialize the delays array
    delays = [0] * M
    delays[0] = X1  # Set the delay for the first train

    # Process the trains
    for i in range(M):
        A_i, B_i, S_i, T_i = trains[i]
        current_delay = delays[i]
        
        # Check for all possible transfers from B_i
        for A_j, S_j, T_j, j in graph[A_i]:
            if T_i + current_delay > S_j + delays[j]:
                # Calculate the required delay for train j
                required_delay = (T_i + current_delay) - S_j
                delays[j] = max(delays[j], required_delay)

    return delays[1:]  # We return delays for trains 2 to M

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M, X1 = map(int, data[0].split())
trains = [tuple(map(int, line.split())) for line in data[1:M+1]]

# Get the result
result = find_delays(N, M, X1, trains)

# Print the result
print(" ".join(map(str, result)))