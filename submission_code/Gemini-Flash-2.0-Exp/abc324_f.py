def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))

    max_ratio = 0.0
    
    def find_paths(start, end, current_path, current_beauty, current_cost):
        nonlocal max_ratio
        
        if start == end:
            if current_cost > 0:
                ratio = current_beauty / current_cost
                max_ratio = max(max_ratio, ratio)
            return
        
        for u, v, b, c in edges:
            if u == start and v not in current_path:
                find_paths(v, end, current_path + [v], current_beauty + b, current_cost + c)

    find_paths(1, n, [1], 0, 0)
    print(max_ratio)

solve()