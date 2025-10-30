# YOUR CODE HERE
def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    min_cost = float('inf')

    for i in range(1 << m):
        cost = 0
        num_edges = 0
        selected_edges = []
        for j in range(m):
            if (i >> j) & 1:
                u, v, w = edges[j]
                cost = (cost + w) % k
                num_edges += 1
                selected_edges.append((u, v))

        if num_edges != n - 1:
            continue

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px = find(x)
            py = find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        num_unions = 0
        temp_parent = list(range(n))
        for u, v in selected_edges:
            if union(u, v):
                num_unions += 1
        
        if num_unions == n - 1:
            min_cost = min(min_cost, cost)

    print(min_cost)

solve()