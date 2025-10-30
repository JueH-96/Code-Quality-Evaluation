import itertools

def min_cost_to_isomorphic(N, M_G, edges_G, M_H, edges_H, A):
    # Create adjacency matrices for graphs G and H
    adj_G = [[0] * N for _ in range(N)]
    adj_H = [[0] * N for _ in range(N)]

    for u, v in edges_G:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1

    for a, b in edges_H:
        adj_H[a-1][b-1] = adj_H[b-1][a-1] = 1

    # Function to check if two graphs are isomorphic
    def are_isomorphic(adj1, adj2, perm):
        for i in range(N):
            for j in range(N):
                if adj1[i][j] != adj2[perm[i]][perm[j]]:
                    return False
        return True

    # Generate all permutations of vertices
    permutations = list(itertools.permutations(range(N)))

    min_cost = float('inf')

    # Check each permutation to find the minimum cost
    for perm in permutations:
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if adj_G[i][j] != adj_H[perm[i]][perm[j]]:
                    cost += A[i][j]
        if are_isomorphic(adj_G, adj_H, perm):
            min_cost = min(min_cost, cost)

    return min_cost

# Read input
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M_G = int(data[index])
index += 1
edges_G = []
for _ in range(M_G):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    edges_G.append((u, v))

M_H = int(data[index])
index += 1
edges_H = []
for _ in range(M_H):
    a = int(data[index])
    index += 1
    b = int(data[index])
    index += 1
    edges_H.append((a, b))

A = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i+1, N):
        A[i][j] = int(data[index])
        index += 1

# Calculate and print the result
result = min_cost_to_isomorphic(N, M_G, edges_G, M_H, edges_H, A)
print(result)