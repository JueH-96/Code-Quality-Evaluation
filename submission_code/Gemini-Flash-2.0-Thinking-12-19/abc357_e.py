def solve():
    n = int(input())
    a = list(map(int, input().split()))
    total_reachable_pairs = 0
    for start_vertex in range(1, n + 1):
        reachable_vertices_set = set()
        current_vertex = start_vertex
        while current_vertex not in reachable_vertices_set:
            reachable_vertices_set.add(current_vertex)
            current_vertex = a[current_vertex - 1]
        total_reachable_pairs += len(reachable_vertices_set)
    print(total_reachable_pairs)

if __name__ == '__main__':
    solve()