import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Building the adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Function to find the minimum number of operations to delete vertex 1
def min_operations_to_delete_vertex_1(N, adj):
    # BFS to calculate the distance from node 1 to all other nodes
    queue = deque([1])
    distance = [-1] * (N + 1)
    distance[1] = 0
    
    while queue:
        node = queue.popleft()
        current_distance = distance[node]
        for neighbor in adj[node]:
            if distance[neighbor] == -1:  # Not visited
                distance[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    # The minimum number of operations to delete vertex 1 is the maximum distance
    # from vertex 1 to any leaf node
    max_distance = 0
    for i in range(1, N + 1):
        if len(adj[i]) == 1:  # It's a leaf if it has only one connection
            max_distance = max(max_distance, distance[i])
    
    return max_distance

# Output the result
print(min_operations_to_delete_vertex_1(N, adj))