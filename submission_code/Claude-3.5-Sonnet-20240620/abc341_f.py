# YOUR CODE HERE
from collections import defaultdict
import sys

def read_input():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))
    return N, graph, W, A

def solve(N, graph, W, A):
    operations = 0
    pieces = sum(A)

    while pieces > 0:
        for i in range(1, N + 1):
            if A[i - 1] > 0:
                A[i - 1] -= 1
                pieces -= 1
                operations += 1

                adjacent_sum = 0
                for neighbor in graph[i]:
                    if adjacent_sum + W[neighbor - 1] < W[i - 1]:
                        adjacent_sum += W[neighbor - 1]
                        A[neighbor - 1] += 1
                        pieces += 1

    return operations

N, graph, W, A = read_input()
result = solve(N, graph, W, A)
print(result)