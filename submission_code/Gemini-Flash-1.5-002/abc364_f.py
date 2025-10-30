import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    edges = []
    for i in range(q):
        l, r, c = map(int, sys.stdin.readline().split())
        for j in range(l, r + 1):
            edges.append((n + i + 1, j, c))

    parent = list(range(n + q + 1))
    rank = [0] * (n + q + 1)

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

    edges.sort(key=lambda x: x[2])
    
    mst_cost = 0
    num_edges = 0
    for u, v, cost in edges:
        if union(u, v):
            mst_cost += cost
            num_edges += 1

    root = find(1)
    connected = True
    for i in range(1, n + q + 1):
        if find(i) != root:
            connected = False
            break

    if connected:
        print(mst_cost)
    else:
        print(-1)

solve()