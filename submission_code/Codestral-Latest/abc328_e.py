import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
for i in range(M):
    u = int(data[3 + 3 * i])
    v = int(data[4 + 3 * i])
    w = int(data[5 + 3 * i])
    edges.append((u, v, w))

# Function to find the parent of a node using path compression
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

# Function to union two sets using rank
def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

# Kruskal's algorithm to find the Minimum Spanning Tree
def kruskal(edges, N):
    result = []
    i = 0
    e = 0
    edges = sorted(edges, key=lambda item: item[2])
    parent = []
    rank = []

    for node in range(N + 1):
        parent.append(node)
        rank.append(0)

    while e < N - 1:
        u, v, w = edges[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append((u, v, w))
            union(parent, rank, x, y)

    return result

# Generate all possible spanning trees
def generate_spanning_trees(edges, N):
    mst_edges = kruskal(edges, N)
    mst_weight = sum(w % K for _, _, w in mst_edges) % K
    return mst_weight

# Main function to find the minimum cost of a spanning tree
def main():
    min_cost = generate_spanning_trees(edges, N)
    print(min_cost)

if __name__ == "__main__":
    main()