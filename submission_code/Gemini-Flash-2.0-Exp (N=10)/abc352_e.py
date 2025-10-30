def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        line = list(map(int, input().split()))
        k = line[0]
        c = line[1]
        nodes = line[2:]
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((nodes[i], nodes[j], c))

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

    if num_edges != n - 1:
        print("-1")
    else:
        print(mst_weight)

solve()