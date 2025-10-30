def solve():
    n = int(input())
    a = list(map(int, input().split()))

    for start_node in range(1, n + 1):
        path = []
        current_node = start_node
        visited_nodes_in_path = set()

        while current_node not in visited_nodes_in_path:
            path.append(current_node)
            visited_nodes_in_path.add(current_node)
            current_node = a[current_node - 1]

        if current_node in visited_nodes_in_path:
            cycle_start_index = -1
            for i in range(len(path)):
                if path[i] == current_node:
                    cycle_start_index = i
                    break
            cycle = path[cycle_start_index:]
            print(len(cycle))
            print(*(cycle))
            return

solve()