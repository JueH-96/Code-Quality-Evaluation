import sys
input = sys.stdin.read

def find(parent, x):
    if parent[x] == x:
        return x
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

def kruskal(N, edges):
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    mst_weight = 0
    edges_used = 0
    for weight, u, v in sorted(edges):
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_weight += weight
            edges_used += 1
            if edges_used == N - 1:
                break
    if edges_used == N - 1:
        return mst_weight
    else:
        return -1

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    edges = []
    for _ in range(M):
        K = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        S = [int(data[index + i]) for i in range(K)]
        index += K
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((C, S[i], S[j]))
    result = kruskal(N, edges)
    print(result)

if __name__ == "__main__":
    main()