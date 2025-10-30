import sys
from collections import defaultdict, deque

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
for i in range(1, N):
    U, V, L = int(data[3*i-2]), int(data[3*i-1]), int(data[3*i])
    edges.append((U, V, L))

# Build the tree
tree = defaultdict(list)
for U, V, L in edges:
    tree[U].append((V, L))
    tree[V].append((U, L))

# Function to find the diameter of the tree
def find_diameter():
    def bfs(start):
        queue = deque([start])
        dist = [-1] * (N + 1)
        dist[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor, length in tree[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + length
                    queue.append(neighbor)
        return dist

    dist1 = bfs(1)
    node1 = dist1.index(max(dist1))
    dist2 = bfs(node1)
    node2 = dist2.index(max(dist2))
    return max(dist2)

# Function to find the longest path passing through a given node
def longest_path_through_node(node):
    def dfs(start, parent):
        max_dist = 0
        for neighbor, length in tree[start]:
            if neighbor != parent:
                dist = dfs(neighbor, start) + length
                max_dist = max(max_dist, dist)
        return max_dist

    return dfs(node, -1)

# Calculate the diameter
diameter = find_diameter()

# Calculate the result for each K
results = []
for K in range(1, N + 1):
    if K == 1:
        results.append(diameter)
    elif K == 2:
        results.append(diameter + 2 * longest_path_through_node(1))
    else:
        results.append(diameter + 2 * longest_path_through_node(1))

# Print the results
for result in results:
    print(result)