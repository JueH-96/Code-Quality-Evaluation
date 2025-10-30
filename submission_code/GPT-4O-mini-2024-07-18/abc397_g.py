def max_shortest_distance(N, M, K, edges):
    from itertools import combinations
    from collections import defaultdict, deque

    # Create a graph representation
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    def bfs(start, end, weights):
        # Perform BFS to find the shortest path from start to end
        queue = deque([start])
        distances = {start: 0}
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            for neighbor in graph[current]:
                weight = weights.get((current, neighbor), 0)
                if neighbor not in distances:
                    distances[neighbor] = current_distance + weight + 1
                    queue.append(neighbor)
        
        return distances.get(end, float('inf'))

    max_distance = 0

    # Generate all combinations of K edges to weight as 1
    for edges_to_weight in combinations(range(M), K):
        weights = {}
        for i in range(M):
            u, v = edges[i]
            if i in edges_to_weight:
                weights[(u, v)] = 1  # Weight this edge as 1
            else:
                weights[(u, v)] = 0  # Weight this edge as 0
        
        # Calculate the shortest distance from 1 to N
        distance = bfs(1, N, weights)
        max_distance = max(max_distance, distance)

    return max_distance

import sys
input = sys.stdin.read
data = input().splitlines()

# Read input
N, M, K = map(int, data[0].split())
edges = [tuple(map(int, line.split())) for line in data[1:M+1]]

# Get the result
result = max_shortest_distance(N, M, K, edges)

# Print the result
print(result)