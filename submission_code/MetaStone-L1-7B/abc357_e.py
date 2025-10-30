def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    visited = [False] * (N + 1)
    reachable_size = [0] * (N + 1)
    
    for u in range(1, N + 1):
        if not visited[u]:
            path = []
            current = u
            while True:
                if visited[current]:
                    if current in path:
                        idx = path.index(current)
                        cycle = path[idx:]
                        cycle_len = len(cycle)
                        for node in cycle:
                            visited[node] = True
                            reachable_size[node] = cycle_len
                        for node in path[:idx]:
                            visited[node] = True
                            next_node = a[node - 1]
                            reachable_size[node] = 1 + reachable_size[next_node]
                    else:
                        for node in path:
                            visited[node] = True
                            next_node = a[node - 1]
                            reachable_size[node] = 1 + reachable_size[next_node]
                    break
                visited[current] = True
                path.append(current)
                current = a[current - 1]
    
    sum_reachable = sum(reachable_size[1:])
    print(sum_reachable)

if __name__ == "__main__":
    main()