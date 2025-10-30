import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
index = 1

# Initialize the adjacency list and in-degree counter
adj_list = [[] for _ in range(N+1)]
in_degree = [0] * (N+1)

# Parse the input data
for i in range(1, N+1):
    C_i = int(data[index])
    index += 1
    for j in range(C_i):
        P_ij = int(data[index])
        index += 1
        adj_list[P_ij].append(i)
        in_degree[i] += 1

# Topological sort using Kahn's algorithm
from collections import deque

queue = deque()
for i in range(1, N+1):
    if in_degree[i] == 0:
        queue.append(i)

order = []
while queue:
    node = queue.popleft()
    order.append(node)
    for neighbor in adj_list[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

# Find the minimum set of books to read to read book 1
visited = [False] * (N+1)
min_books = []

def dfs(book):
    if visited[book]:
        return
    visited[book] = True
    for neighbor in adj_list[book]:
        dfs(neighbor)
    if book != 1:
        min_books.append(book)

dfs(1)
min_books.reverse()

# Print the result
print(" ".join(map(str, min_books)))