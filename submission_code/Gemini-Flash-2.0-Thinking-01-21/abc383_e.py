import collections

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    a_seq = list(map(int, input().split()))
    b_seq = list(map(int, input().split()))
    
    def get_min_path_weight(start_node, end_node):
        if start_node == end_node:
            return 0
        min_weights = {i: float('inf') for i in range(1, n + 1)}
        min_weights[start_node] = 0
        pq = [(0, start_node)]
        
        while pq:
            current_weight, u = min(pq, key=lambda x: x[0])
            pq.remove((current_weight, u))
            if current_weight > min_weights[u]:
                continue
            for (v1, v2), weight in edges:
                v = -1
                if v1 == u:
                    v = v2
                elif v2 == u:
                    v = v1
                else:
                    continue
                path_weight = max(current_weight, weight)
                if path_weight < min_weights[v]:
                    min_weights[v] = path_weight
                    pq.append((path_weight, v))
                    
        return min_weights[end_node]

    cost_matrix = []
    for i in range(k):
        row_costs = []
        for j in range(k):
            cost = get_min_path_weight(a_seq[i], b_seq[j])
            row_costs.append(cost)
        cost_matrix.append(row_costs)
        
    import itertools
    
    min_total_cost = float('inf')
    permutations_b = list(itertools.permutations(b_seq))
    
    for perm_b in permutations_b:
        current_total_cost = 0
        for i in range(k):
            cost = get_min_path_weight(a_seq[i], perm_b[i])
            current_total_cost += cost
        min_total_cost = min(min_total_cost, current_total_cost)
        
    print(min_total_cost)

if __name__ == '__main__':
    solve()