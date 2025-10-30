# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Initialize the adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u-1].append(v-1)
        adj[v-1].append(u-1)

    # Initialize the XOR values
    xor_values = [0] * N

    # For each vertex, calculate the XOR of its adjacent vertices
    for u in range(N):
        for v in adj[u]:
            xor_values[u] ^= v+1

    # If there is a vertex with a degree of at least 1 and its XOR value is not 0,
    # then there is no solution
    for u in range(N):
        if len(adj[u]) > 0 and xor_values[u] != 0:
            print("No")
            return

    # Otherwise, print the XOR values
    print("Yes")
    print(" ".join(map(str, xor_values)))

solve()