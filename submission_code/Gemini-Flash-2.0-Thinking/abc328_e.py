import itertools

def solve():
    n, m, k = map(int, input().split())
    edges_data = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges_data.append((u, v, w))

    min_cost = float('inf')

    for combo_indices in itertools.combinations(range(m), n - 1):
        combo_edges_indices = list(combo_indices)
        combo_edges = [edges_data[i] for i in combo_edges_indices]

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

        num_components = n
        for u, v, _ in combo_edges:
            if union(u, v):
                num_components -= 1

        roots = set()
        for i in range(1, n + 1):
            roots.add(find(i))

        if len(roots) == 1:
            current_cost = sum(w for _, _, w in combo_edges) % k
            min_cost = min(min_cost, current_cost)

    print(min_cost)

solve()