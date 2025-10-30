# YOUR CODE HERE
def find_strongest_programmer(N, M, superiority_info):
    from collections import defaultdict, deque

    # Create adjacency list for the directed graph
    graph = defaultdict(list)
    in_degree = [0] * (N + 1)

    for A, B in superiority_info:
        graph[A].append(B)
        in_degree[B] += 1

    # Topological sort using Kahn's algorithm
    queue = deque()
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            queue.append(i)

    topological_order = []
    while queue:
        if len(queue) > 1:
            return -1  # More than one node with in-degree 0 means multiple possible strongest
        node = queue.popleft()
        topological_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topological_order) != N:
        return -1  # If topological sort doesn't include all nodes, there's a cycle or missing info

    return topological_order[0]

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
superiority_info = [(int(data[i*2+2]), int(data[i*2+3])) for i in range(M)]

# Find and print the strongest programmer
print(find_strongest_programmer(N, M, superiority_info))