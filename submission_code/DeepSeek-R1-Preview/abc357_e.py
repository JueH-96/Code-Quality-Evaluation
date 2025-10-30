def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    for i in range(N):
        a[i] -= 1  # Convert to 0-based

    visited = [False] * N
    on_stack = [False] * N
    stack = []
    index = [-1] * N
    cycles = []

    # Step 1: Detect all cycles
    for u in range(N):
        if not visited[u]:
            path = []
            current = u
            while True:
                if visited[current]:
                    if on_stack[current]:
                        # Found a cycle
                        idx = path.index(current)
                        cycle = path[idx:]
                        cycles.append(cycle)
                    break
                visited[current] = True
                on_stack[current] = True
                path.append(current)
                current = a[current]
            # Reset on_stack for the path
            for node in path:
                on_stack[node] = False

    # Step 2: Compute cycle sizes
    cycle_size = [0] * N
    for cycle in cycles:
        sz = len(cycle)
        for node in cycle:
            cycle_size[node] = sz

    # Step 3: Compute reachable count for each node
    reachable_count = [0] * N
    processed = [False] * N

    for u in range(N):
        if not processed[u]:
            path = []
            current = u
            while True:
                if processed[current] or cycle_size[current] > 0:
                    # Process the path
                    if cycle_size[current] > 0:
                        # Nodes in path are leading into a cycle
                        sz = cycle_size[current]
                        for i in range(len(path)):
                            node = path[i]
                            reachable_count[node] = (len(path) - i) + sz
                            processed[node] = True
                    else:
                        # Nodes in path are leading into a processed tree node
                        rc = reachable_count[current]
                        for i in range(len(path)):
                            node = path[i]
                            reachable_count[node] = (len(path) - i) + rc
                            processed[node] = True
                    break
                if processed[current]:
                    break
                path.append(current)
                processed[current] = True  # Mark as being processed to avoid loops
                current = a[current]

    # For nodes in cycles, set their reachable count
    for cycle in cycles:
        sz = len(cycle)
        for node in cycle:
            reachable_count[node] = sz

    # Now, sum all reachable counts
    total = sum(reachable_count)
    print(total)

if __name__ == '__main__':
    main()