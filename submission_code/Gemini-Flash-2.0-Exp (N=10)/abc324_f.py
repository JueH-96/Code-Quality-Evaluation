def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))

    def calculate_ratio(path_edges):
        total_beauty = 0
        total_cost = 0
        for edge_index in path_edges:
            u, v, b, c = edges[edge_index]
            total_beauty += b
            total_cost += c
        if total_cost == 0:
            return 0
        return total_beauty / total_cost
    
    max_ratio = 0
    
    def find_paths(current_node, current_path_edges):
        nonlocal max_ratio
        if current_node == n:
            max_ratio = max(max_ratio, calculate_ratio(current_path_edges))
            return
        
        for i, (u, v, _, _) in enumerate(edges):
            if u == current_node:
                find_paths(v, current_path_edges + [i])

    find_paths(1, [])
    print(max_ratio)

solve()