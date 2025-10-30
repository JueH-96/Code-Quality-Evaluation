from collections import defaultdict
import sys

# Read input
N, M, X_1 = map(int, input().split())
trains = [tuple(map(int, input().split())) for _ in range(M)]

# Create a graph to represent connections between trains
graph = defaultdict(list)
for i, (A, B, S, T) in enumerate(trains):
    graph[B].append((i, S, T))

# Sort the graph based on arrival times
for k in graph:
    graph[k].sort(key=lambda x: x[2])

# Initialize delays and set the delay of the first train
delays = [0] * M
delays[0] = X_1

# Process each train to find the minimum delays
for i in range(M):
    A, B, S, T = trains[i]
    for j, S_j, T_j in graph[A]:
        if T + delays[i] <= S_j:
            delays[j] = max(delays[j], T + delays[i] - S_j)

# Output the delays except for the first train
print(' '.join(map(str, delays[1:])))