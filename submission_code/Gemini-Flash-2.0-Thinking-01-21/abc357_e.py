def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_reachable_pairs = 0
    for start_vertex in range(1, n + 1):
        reachable_vertices_path = []
        current_vertex = start_vertex
        path_set = set()
        while current_vertex not in path_set:
            path_set.add(current_vertex)
            reachable_vertices_path.append(current_vertex)
            current_vertex = a[current_vertex - 1]
        reachable_count = len(reachable_vertices_path)
        total_reachable_pairs += reachable_count
        
    print(total_reachable_pairs)

if __name__ == '__main__':
    solve()