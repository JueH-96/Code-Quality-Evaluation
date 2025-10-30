import collections

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    a_vals = list(map(int, input().split()))
    b_vals = list(map(int, input().split()))
    
    min_path_weights = collections.defaultdict(lambda: float('inf'))
    for i in range(1, n + 1):
        min_path_weights[(i, i)] = 0
        
    sorted_edges = sorted(edges, key=lambda x: x[1])
    parent = list(range(n + 1))
    
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]
        
    def unite_sets(u, v):
        u = find_set(u)
        v = find_set(v)
        if u != v:
            parent[u] = v
            return True
        return False
        
    components = [set([i]) for i in range(n + 1)]
    components[0] = set()

    for (u, v), weight in sorted_edges:
        root_u = find_set(u)
        root_v = find_set(v)
        if root_u != root_v:
            component_u = set()
            component_v = set()
            for i in range(1, n + 1):
                if find_set(i) == root_u:
                    component_u.add(i)
                if find_set(i) == root_v:
                    component_v.add(i)
                    
            for u_comp in component_u:
                for v_comp in component_v:
                    if min_path_weights[(u_comp, v_comp)] == float('inf'):
                        min_path_weights[(u_comp, v_comp)] = weight
                        min_path_weights[(v_comp, u_comp)] = weight
                        
            unite_sets(u, v)

    cost_matrix = []
    for i in range(k):
        row_costs = []
        for j in range(k):
            row_costs.append(min_path_weights[(a_vals[i], b_vals[j])])
        cost_matrix.append(row_costs)
        
    min_sum_weight = 0
    assigned_cols = [False] * k
    
    for _ in range(k):
        min_cost_val = float('inf')
        best_row_index = -1
        best_col_index = -1
        for i in range(k):
            if best_row_index != -1:
                break
            for j in range(k):
                if not assigned_cols[j]:
                    if cost_matrix[i][j] < min_cost_val:
                        min_cost_val = cost_matrix[i][j]
                        best_row_index = i
                        best_col_index = j
                        
        min_sum_weight += min_cost_val
        assigned_cols[best_col_index] = True
        
    print(min_sum_weight)

if __name__ == '__main__':
    solve()