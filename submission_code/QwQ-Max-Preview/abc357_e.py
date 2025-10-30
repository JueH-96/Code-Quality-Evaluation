def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    visited = [0] * (n + 1)  # 0=unvisited, 1=visiting, 2=processed
    in_cycle = [False] * (n + 1)
    cycle_size = [0] * (n + 1)
    
    for u in range(1, n + 1):
        if visited[u] == 0:
            path = []
            current = u
            while True:
                if visited[current] == 1:
                    # Found a cycle
                    idx = path.index(current)
                    cycle = path[idx:]
                    sz = len(cycle)
                    # Mark cycle nodes as processed and set cycle info
                    for v in cycle:
                        in_cycle[v] = True
                        cycle_size[v] = sz
                        visited[v] = 2
                    # Reset visited for nodes before the cycle
                    for v in path[:idx]:
                        visited[v] = 0
                    break
                if visited[current] == 2:
                    break
                visited[current] = 1
                path.append(current)
                current = a[current - 1]  # a is 1-based
    
    # Step 2: Process nodes not in cycles
    memo = [(0, 0)] * (n + 1)  # (cycle_size, distance)
    processed = [False] * (n + 1)
    
    for u in range(1, n + 1):
        if not processed[u] and not in_cycle[u]:
            chain = []
            current = u
            while True:
                if processed[current] or in_cycle[current]:
                    break
                chain.append(current)
                processed[current] = True
                current = a[current - 1]
            # Process the chain in reverse
            for v in reversed(chain):
                next_v = a[v - 1]
                if in_cycle[next_v]:
                    cs = cycle_size[next_v]
                    dist = 1
                else:
                    cs, next_dist = memo[next_v]
                    dist = 1 + next_dist
                memo[v] = (cs, dist)
    
    total = 0
    for v in range(1, n + 1):
        if in_cycle[v]:
            total += cycle_size[v]
        else:
            total += memo[v][0] + memo[v][1]
    
    print(total)

if __name__ == '__main__':
    main()