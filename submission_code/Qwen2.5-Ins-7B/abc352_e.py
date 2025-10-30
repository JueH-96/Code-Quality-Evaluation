# YOUR CODE HERE
from collections import defaultdict
from heapq import heappop, heappush
from sys import stdin

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

def kruskal(n, edges):
    parent = list(range(n))
    rank = [0] * n
    mst = []
    edges.sort()
    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst.append((weight, u, v))
    return mst

def main():
    N, M = map(int, stdin.readline().split())
    edges = []
    for _ in range(M):
        K, C = map(int, stdin.readline().split())
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((C, A[i] - 1, A[j] - 1))
        A = list(map(int, stdin.readline().split()))
    parent = list(range(N))
    rank = [0] * N
    components = set(range(N))
    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            components.discard(find(parent, u))
    if len(components) > 1:
        print(-1)
    else:
        mst = kruskal(N, edges)
        print(sum(weight for weight, _, _ in mst))

if __name__ == "__main__":
    main()