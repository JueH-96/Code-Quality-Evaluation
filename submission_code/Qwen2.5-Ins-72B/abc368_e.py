# YOUR CODE HERE
import sys
from heapq import heappop, heappush

def solve(N, M, X1, trains):
    # Initialize the graph
    graph = [[] for _ in range(M + 1)]
    for i in range(M):
        A, B, S, T = trains[i]
        for j in range(M):
            if B == trains[j][0] and T <= trains[j][2]:
                graph[i + 1].append((j + 1, trains[j][2] - T))
    
    # Initialize the distances
    dist = [float('inf')] * (M + 1)
    dist[1] = X1
    
    # Dijkstra's algorithm
    pq = [(X1, 1)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
    
    # Output the results
    return [dist[i] - X1 for i in range(2, M + 1)]

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
X1 = int(data[2])
trains = [(int(data[3 * i + 3]), int(data[3 * i + 4]), int(data[3 * i + 5]), int(data[3 * i + 6])) for i in range(M)]

# Solve and print the result
result = solve(N, M, X1, trains)
print(" ".join(map(str, result)))