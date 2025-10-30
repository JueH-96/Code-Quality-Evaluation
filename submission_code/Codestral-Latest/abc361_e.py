import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
for i in range(1, len(data), 3):
    A = int(data[i]) - 1
    B = int(data[i + 1]) - 1
    C = int(data[i + 2])
    edges.append((A, B, C))

# Build the graph
graph = defaultdict(list)
for A, B, C in edges:
    graph[A].append((B, C))
    graph[B].append((A, C))

# Function to find the diameter of the tree
def find_diameter():
    def bfs(start):
        queue = deque([start])
        visited = [-1] * N
        visited[start] = 0
        max_dist = 0
        farthest_node = start

        while queue:
            node = queue.popleft()
            for neighbor, weight in graph[node]:
                if visited[neighbor] == -1:
                    visited[neighbor] = visited[node] + weight
                    queue.append(neighbor)
                    if visited[neighbor] > max_dist:
                        max_dist = visited[neighbor]
                        farthest_node = neighbor

        return farthest_node, max_dist

    node1, _ = bfs(0)
    node2, diameter = bfs(node1)
    return diameter

# Calculate the minimum travel distance
diameter = find_diameter()
total_distance = diameter + sum(C for _, _, C in edges)

print(total_distance)