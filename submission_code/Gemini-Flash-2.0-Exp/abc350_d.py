def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u - 1, v - 1))

    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    for u, v in edges:
        union(u, v)

    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = 0
        components[root] += 1

    ans = 0
    for root in components:
        size = components[root]
        num_edges = 0
        for u, v in edges:
            if find(u) == root:
                num_edges += 1
        
        num_edges //= 2
        
        ans += (size * (size - 1) // 2) - num_edges

    print(ans)

solve()