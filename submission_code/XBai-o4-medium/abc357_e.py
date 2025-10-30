import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    a = [0] + a  # 1-based indexing

    status = [0] * (N + 1)  # 0: unvisited, 1: processing, 2: done
    cycle_size = [0] * (N + 1)
    distance_to_cycle = [0] * (N + 1)

    for i in range(1, N + 1):
        if status[i] == 0:
            path = []
            current = i
            while True:
                path.append(current)
                status[current] = 1
                next_node = a[current]
                if status[next_node] == 0:
                    current = next_node
                else:
                    if status[next_node] == 1:
                        # Found a cycle
                        pos = path.index(next_node)
                        cycle_length = len(path) - pos
                        # Process cycle nodes
                        for j in range(pos, len(path)):
                            node = path[j]
                            cycle_size[node] = cycle_length
                            distance_to_cycle[node] = 0
                        # Process the rest of the path
                        for j in range(pos - 1, -1, -1):
                            node = path[j]
                            next_n = a[node]
                            cycle_size[node] = cycle_size[next_n]
                            distance_to_cycle[node] = distance_to_cycle[next_n] + 1
                    else:
                        # next_node is already processed (status 2)
                        # Process all nodes in path in reverse order
                        for j in range(len(path) - 1, -1, -1):
                            node = path[j]
                            next_n = a[node]
                            cycle_size[node] = cycle_size[next_n]
                            distance_to_cycle[node] = distance_to_cycle[next_n] + 1
                    # Mark all nodes in path as done
                    for node in path:
                        status[node] = 2
                    break

    ans = 0
    for u in range(1, N + 1):
        ans += cycle_size[u] + distance_to_cycle[u]
    print(ans)

if __name__ == '__main__':
    main()