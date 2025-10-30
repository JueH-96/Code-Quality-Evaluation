def solve():
    n = int(input())
    a = list(map(int, input().split()))
    a = [x - 1 for x in a]  # 0-based indexing

    visited = {}
    path = []
    current_node = 0
    while current_node not in visited:
        visited[current_node] = len(path)
        path.append(current_node)
        current_node = a[current_node]

    cycle_start_index = visited[current_node]
    cycle_nodes = path[cycle_start_index:]
    cycle_nodes_1_based = [node + 1 for node in cycle_nodes]

    print(len(cycle_nodes_1_based))
    print(*cycle_nodes_1_based)

solve()