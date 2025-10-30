import sys
from heapq import heappop, heappush

def kruskal(n, edges):
    parent = list(range(n + 1))
    rank = [0] * (n + 1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    edges.sort(key=lambda x: x[2])
    mst_cost = 0
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            mst_cost += w

    # Check if the graph is connected
    root = find(1)
    for i in range(2, n + 1):
        if find(i) != root:
            return -1

    return mst_cost

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    Q = int(data[index + 1])
    index += 2

    edges = []
    for _ in range(Q):
        L = int(data[index])
        R = int(data[index + 1])
        C = int(data[index + 2])
        index += 3
        for j in range(L, R + 1):
            edges.append((N + _, j, C))

    result = kruskal(N + Q, edges)
    print(result)

if __name__ == "__main__":
    main()