def solve():
    n, q = map(int, input().split())
    edges = []
    for i in range(q):
        l, r, c = map(int, input().split())
        edges.append((l, r, c, n + i + 1))

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

    all_edges = []
    for l, r, c, node in edges:
        for i in range(l, r + 1):
            all_edges.append((c, node, i))
    
    all_edges.sort()

    mst_cost = 0
    num_edges = 0
    for cost, u, v in all_edges:
        if union(u, v):
            mst_cost += cost
            num_edges += 1

    if num_edges == n + q - 1:
        
        connected = True
        root = find(1)
        for i in range(2, n + q + 1):
            if find(i) != root:
                connected = False
                break
        if connected:
            print(mst_cost)
        else:
            print("-1")
    else:
        print("-1")

solve()