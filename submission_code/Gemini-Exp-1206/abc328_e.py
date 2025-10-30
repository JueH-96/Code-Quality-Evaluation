def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    min_cost = float('inf')

    for i in range(1 << m):
        subset = []
        for j in range(m):
            if (i >> j) & 1:
                subset.append(edges[j])

        if len(subset) != n - 1:
            continue

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

        is_spanning_tree = True
        cost = 0
        num_edges = 0
        
        for u, v, w in subset:
            if union(u, v):
                cost = (cost + w) % k
                num_edges += 1
            else:
                is_spanning_tree = False
                break

        if is_spanning_tree and num_edges == n-1:
            
            
            root_count = 0
            for i in range(1,n+1):
                if parent[i] == i:
                    root_count += 1
            if root_count == 1:
                min_cost = min(min_cost, cost)

    print(min_cost)

solve()