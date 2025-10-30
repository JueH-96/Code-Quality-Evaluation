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
    mst_cost = 0
    edges_used = 0
    for cost, u, v in edges:
        if union(parent, rank, u, v):
            mst_cost += cost
            edges_used += 1
            if edges_used == N:
                break
    return mst_cost if edges_used == N else -1

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    Q = int(input[index + 1])
    index += 2
    edges = []
    for _ in range(Q):
        L = int(input[index])
        R = int(input[index + 1])
        C = int(input[index + 2])
        index += 3
        for j in range(L, R + 1):
            edges.append((C, j, N + _ + 1))
    result = kruskal(N + Q, edges)
    print(result)

if __name__ == "__main__":
    main()