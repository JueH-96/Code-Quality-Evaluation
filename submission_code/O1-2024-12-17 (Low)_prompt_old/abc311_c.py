def solve():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Convert A to 0-based for convenience
    A = [x - 1 for x in A]

    visited = [0]*N  # 0 = unvisited, 1 = in current chain, 2 = processed

    for start in range(N):
        if visited[start] == 0:
            path = []
            current = start
            while visited[current] == 0:
                visited[current] = 1
                path.append(current)
                current = A[current]
            if visited[current] == 1:
                # We found a cycle
                # Find where current first appeared in path
                idx = path.index(current)
                cycle = path[idx:]
                # Print the cycle in 1-based indexing
                print(len(cycle))
                print(' '.join(str(v+1) for v in cycle))
                return

            # Mark all nodes in path as fully processed (2)
            for node in path:
                visited[node] = 2

# Call solve()
solve()