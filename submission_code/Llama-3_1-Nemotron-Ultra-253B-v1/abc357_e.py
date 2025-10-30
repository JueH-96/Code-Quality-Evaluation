def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    a = list(map(int, input[1:N+1]))
    visited = [False] * (N + 1)
    cycle_size_for_node = [0] * (N + 1)
    total = 0

    for i in range(1, N + 1):
        if not visited[i]:
            path = []
            pos_map = {}
            current = i
            while True:
                if visited[current]:
                    if current in pos_map:
                        idx = pos_map[current]
                        cycle_size = len(path) - idx
                        total += cycle_size * cycle_size
                        m = idx
                        sum_tree = m * (m + 1) // 2 + m * cycle_size
                        total += sum_tree
                        for node in path[idx:]:
                            cycle_size_for_node[node] = cycle_size
                        break
                    else:
                        m = len(path)
                        cycle_size = cycle_size_for_node[current]
                        sum_tree = m * (m + 1) // 2 + m * cycle_size
                        total += sum_tree
                        break
                else:
                    visited[current] = True
                    pos_map[current] = len(path)
                    path.append(current)
                    current = a[current - 1]
    print(total)

if __name__ == "__main__":
    main()