import sys
from collections import defaultdict, deque

# YOUR CODE HERE
def max_operations(N, M, edges, W, A):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    def can_place(x, S):
        return sum(W[y] for y in S) < W[x]
    
    operations = 0
    while True:
        placed = False
        for i in range(1, N + 1):
            if A[i - 1] > 0:
                A[i - 1] -= 1
                placed = True
                S = [j for j in graph[i] if A[j - 1] == 0 and can_place(i, [j])]
                for j in S:
                    A[j - 1] += 1
                operations += 1
                break
        if not placed:
            break
    return operations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
edges = [(int(data[2 + 2 * i]) - 1, int(data[3 + 2 * i]) - 1) for i in range(M)]
W = [int(data[2 + 2 * M + i]) for i in range(N)]
A = [int(data[3 + 2 * M + N + i]) for i in range(N)]

# Solve the problem
result = max_operations(N, M, edges, W, A)

# Print the result
print(result)