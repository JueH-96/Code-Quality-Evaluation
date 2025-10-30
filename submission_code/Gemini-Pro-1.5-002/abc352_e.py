# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        line = list(map(int, sys.stdin.readline().split()))
        k = line[0]
        c = line[1]
        vertices = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                u = vertices[i] - 1
                v = vertices[j] - 1
                adj[u].append((v, c))
                adj[v].append((u, c))

    parent = list(range(n))
    rank = [0] * n

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1
            return True
        return False

    edges = []
    for u in range(n):
        for v, c in adj[u]:
            edges.append((u, v, c))

    edges.sort(key=lambda x: x[2])

    mst_weight = 0
    num_edges = 0
    for u, v, c in edges:
        if union(u, v):
            mst_weight += c
            num_edges += 1

    root = find(0)
    for i in range(1, n):
        if find(i) != root:
            print(-1)
            return

    print(mst_weight)


solve()