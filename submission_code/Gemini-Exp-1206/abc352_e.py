def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        line = list(map(int, input().split()))
        k, c = line[0], line[1]
        a = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((a[i], a[j], c))

    parent = list(range(n + 1))
    rank = [0] * (n + 1)

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
    
    mst_weight = 0
    num_edges = 0
    
    for u, v, w in edges:
        if union(u, v):
            mst_weight += w
            num_edges += 1

    
    
    connected_components = 0
    for i in range(1, n+1):
        if parent[i] == i:
            connected_components += 1
    
    if connected_components > 1:
        print("-1")
    else:
        print(mst_weight)

solve()