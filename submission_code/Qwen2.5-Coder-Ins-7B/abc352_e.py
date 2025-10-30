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

def kruskal(n, edges):
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    mst_weight = 0
    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_weight += weight
    return mst_weight

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2
    
    edges = []
    for _ in range(M):
        K = int(data[index])
        C = int(data[index + 1])
        index += 2
        S = set()
        for _ in range(K):
            S.add(int(data[index]))
            index += 1
        for u in S:
            for v in S:
                if u < v:
                    edges.append((C, u, v))
    
    edges.sort()
    
    if len(set(find(parent, i) for i in range(1, N + 1))) > 1:
        print(-1)
    else:
        mst_weight = kruskal(N, edges)
        print(mst_weight)

if __name__ == "__main__":
    main()