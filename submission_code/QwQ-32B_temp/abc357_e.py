import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    visited = [0] * (N + 1)  # 0: unvisited, 1: visiting, 2: done
    d = [0] * (N + 1)
    c = [0] * (N + 1)

    for u in range(1, N + 1):
        if visited[u] == 0:
            path = []
            pos = {}
            current_node = u
            while True:
                if visited[current_node] == 1:
                    # Found a cycle
                    idx = pos[current_node]
                    cycle_len = len(path) - idx
                    for i in range(len(path)):
                        node = path[i]
                        if i < idx:
                            d[node] = idx - i
                            c[node] = cycle_len
                        else:
                            d[node] = 0
                            c[node] = cycle_len
                    # Mark all nodes in path as visited
                    for node in path:
                        visited[node] = 2
                    break
                elif visited[current_node] == 2:
                    # Current node is already processed
                    cycle_len = c[current_node]
                    dist_current = d[current_node]
                    for i in range(len(path)):
                        node = path[i]
                        distance = (len(path) - i) + dist_current
                        d[node] = distance
                        c[node] = cycle_len
                    # Mark all nodes in path as visited
                    for node in path:
                        visited[node] = 2
                    break
                else:  # unvisited
                    visited[current_node] = 1
                    pos[current_node] = len(path)
                    path.append(current_node)
                    current_node = a[current_node - 1]

    total = 0
    for i in range(1, N + 1):
        total += d[i] + c[i]
    print(total)

if __name__ == "__main__":
    main()