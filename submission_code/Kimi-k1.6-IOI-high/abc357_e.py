def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    a = list(map(int, input[1:N+1]))
    
    visited = [0] * (N + 1)  # 0: unvisited, 1: visiting, 2: visited
    count = [0] * (N + 1)
    
    for u in range(1, N + 1):
        if visited[u] == 0:
            path = []
            pos = dict()
            current = u
            while True:
                if visited[current] == 0:
                    visited[current] = 1
                    path.append(current)
                    pos[current] = len(path) - 1
                    current = a[current - 1]  # a is 0-based in the list, nodes are 1-based
                elif visited[current] == 1:
                    # Found a cycle
                    idx = pos[current]
                    cycle_length = len(path) - idx
                    for node in path[idx:]:
                        visited[node] = 2
                        count[node] = cycle_length
                    # Process nodes before the cycle
                    for i in range(idx - 1, -1, -1):
                        node = path[i]
                        visited[node] = 2
                        next_node = path[i + 1]
                        count[node] = 1 + count[next_node]
                    break
                else:
                    # Current is already visited (state 2)
                    prev_count = count[current]
                    # Process the path in reverse order
                    for i in range(len(path) - 1, -1, -1):
                        node = path[i]
                        visited[node] = 2
                        count[node] = prev_count + 1
                        prev_count = count[node]
                    break
    total = sum(count[1:N+1])
    print(total)

if __name__ == "__main__":
    main()