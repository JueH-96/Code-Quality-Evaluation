import sys
from collections import deque

# Read all input and convert to integers
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
M = data[index]
index += 1

# Create adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    A = data[index]
    index += 1
    B = data[index]
    index += 1
    adj[A].append(B)
    adj[B].append(A)

# Array to keep track of visited nodes
visited = [False] * (N + 1)
sum_binom = 0

# Iterate through all nodes to find connected components
for node in range(1, N + 1):
    if not visited[node]:
        # BFS to find component size
        queue = deque()
        queue.append(node)
        visited[node] = True
        comp_size = 0
        while queue:
            current = queue.popleft()
            comp_size += 1
            for neighbor in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        # Add binomial coefficient for component size
        sum_binom += (comp_size * (comp_size - 1)) // 2

# Calculate the number of added edges
answer = sum_binom - M
print(answer)