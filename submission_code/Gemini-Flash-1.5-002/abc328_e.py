# YOUR CODE HERE
import itertools

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for i in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    min_cost = float('inf')
    for i in range(1 << m):
        num_edges = 0
        cost = 0
        selected_edges = []
        for j in range(m):
            if (i >> j) & 1:
                num_edges += 1
                cost = (cost + edges[j][2]) % k
                selected_edges.append(edges[j])

        if num_edges != n - 1:
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
                return True
            return False

        num_trees = n
        for u, v, w in selected_edges:
            if union(u,v):
                num_trees -=1

        if num_trees == 1:
            min_cost = min(min_cost, cost)

    print(min_cost)

solve()