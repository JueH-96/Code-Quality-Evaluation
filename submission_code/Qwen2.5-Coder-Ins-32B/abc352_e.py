# YOUR CODE HERE
import sys
import heapq

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1
        return True
    return False

def kruskal(N, edges):
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    edges.sort()
    mst_weight = 0
    edges_used = 0

    for weight, u, v in edges:
        if union(parent, rank, u, v):
            mst_weight += weight
            edges_used += 1
            if edges_used == N - 1:
                return mst_weight

    return -1

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    edges = []

    for _ in range(M):
        K = int(input[index])
        C = int(input[index + 1])
        index += 2
        vertices = list(map(int, input[index:index + K]))
        index += K
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((C, vertices[i], vertices[j]))

    result = kruskal(N, edges)
    print(result)

if __name__ == "__main__":
    main()