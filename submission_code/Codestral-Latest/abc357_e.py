import sys
from collections import defaultdict

# Read input
N = int(sys.stdin.readline().strip())
a = list(map(int, sys.stdin.readline().strip().split()))

# Create a graph representation
graph = defaultdict(list)
for i in range(N):
    graph[a[i] - 1].append(i)

# Function to perform DFS and count reachable vertices
def dfs(node, visited):
    stack = [node]
    count = 0
    while stack:
        current = stack.pop()
        if not visited[current]:
            visited[current] = True
            count += 1
            stack.extend(graph[current])
    return count

# Initialize visited array
visited = [False] * N

# Calculate the number of reachable vertices from each vertex
reachable_count = [0] * N
for i in range(N):
    if not visited[i]:
        reachable_count[i] = dfs(i, visited)

# Calculate the total number of pairs (u, v) such that v is reachable from u
total_pairs = 0
for i in range(N):
    total_pairs += reachable_count[a[i] - 1]

# Print the result
print(total_pairs)