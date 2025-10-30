def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    visited = [0] * (N + 1)  # 0=unvisited, 1=in current path, 2=fully processed

    for start in range(1, N + 1):
        if visited[start] != 0:
            continue

        path = []
        path_index = {}
        node = start

        while True:
            visited[node] = 1
            path_index[node] = len(path)
            path.append(node)

            next_node = A[node - 1]
            if visited[next_node] == 0:
                node = next_node
                continue
            elif visited[next_node] == 1:
                # Found a cycle
                start_idx = path_index[next_node]
                cycle = path[start_idx:]
                print(len(cycle))
                print(" ".join(map(str, cycle)))
                return
            else:
                # visited[next_node] == 2, so no cycle continuing from here
                break

        # Mark all in path as fully processed
        for x in path:
            visited[x] = 2

# Do not forget to call main()
if __name__ == "__main__":
    main()