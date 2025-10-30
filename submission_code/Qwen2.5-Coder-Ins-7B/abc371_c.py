import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M_G = int(data[1])
edges_G = []
for i in range(M_G):
    u = int(data[2 + 2 * i]) - 1
    v = int(data[3 + 2 * i]) - 1
    edges_G.append((u, v))

M_H = int(data[2 + 2 * M_G])
edges_H = []
for i in range(M_H):
    a = int(data[3 + 2 * M_G + 2 * i]) - 1
    b = int(data[4 + 2 * M_G + 2 * i]) - 1
    edges_H.append((a, b))

A = []
for i in range(N - 1):
    row = []
    for j in range(i + 1, N):
        row.append(int(data[5 + 2 * M_G + 2 * M_H + i * (N - i - 1) + j - i - 1]))
    A.append(row)

# Create adjacency lists for both graphs
adj_G = defaultdict(list)
for u, v in edges_G:
    adj_G[u].append(v)
    adj_G[v].append(u)

adj_H = defaultdict(list)
for a, b in edges_H:
    adj_H[a].append(b)
    adj_H[b].append(a)

# Function to check if two graphs are isomorphic
def is_isomorphic(adj_G, adj_H):
    if len(adj_G) != len(adj_H):
        return False
    if len(adj_G) == 0:
        return True
    n = len(adj_G)
    for perm in permutations(range(n)):
        if all(len(adj_G[i]) == len(adj_H[perm[i]]) for i in range(n)):
            if all(sorted(adj_G[i]) == sorted(adj_H[perm[i]]) for i in range(n)):
                return True
    return False

# Function to calculate the minimum cost to make the graphs isomorphic
def min_cost_to_isomorphic(adj_G, adj_H, A):
    if is_isomorphic(adj_G, adj_H):
        return 0
    min_cost = float('inf')
    for perm in permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in adj_G and perm[j] not in adj_H[perm[i]]:
                    cost += A[i][j]
                elif (i, j) not in adj_G and perm[j] in adj_H[perm[i]]:
                    cost += A[i][j]
        min_cost = min(min_cost, cost)
    return min_cost

# Calculate and print the minimum cost
print(min_cost_to_isomorphic(adj_G, adj_H, A))