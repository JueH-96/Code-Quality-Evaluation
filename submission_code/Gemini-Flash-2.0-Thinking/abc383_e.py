from scipy.optimize import linear_sum_assignment
import numpy as np

def solve():
    n, m, k = map(int, input().split())
    edges_list = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges_list.append((u, v, w))

    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def find_f(u, v, n, edges):
        sorted_edges = sorted(edges, key=lambda x: x[2])
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

        for weight, u_edge, v_edge in sorted((w, u1, v1) for u1, v1, w in edges):
            union(u_edge, v_edge)
            if find(u) == find(v):
                return weight
        return float('inf')

    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost_matrix[i, j] = find_f(a[i], b[j], n, edges_list)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_sum = cost_matrix[row_ind, col_ind].sum()
    print(int(min_sum))

solve()