def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    min_cost = float('inf')

    for i in range(1 << m):
        if bin(i).count('1') != n - 1:
            continue
        
        selected_edges = []
        for j in range(m):
            if (i >> j) & 1:
                selected_edges.append(edges[j])
        
        if len(selected_edges) != n - 1:
            continue
        
        parent = list(range(n + 1))
        
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
        
        
        for u, v, _ in selected_edges:
            union(u, v)
        
        
        num_components = 0
        for j in range(1, n + 1):
            if parent[j] == j:
                num_components += 1
        
        if num_components != 1:
            continue
        
        current_cost = 0
        for _, _, w in selected_edges:
            current_cost = (current_cost + w) % k
        
        min_cost = min(min_cost, current_cost)
    
    print(min_cost)

solve()