def main():
    import sys
    from collections import defaultdict

    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    edges = list(map(int, input[2:2*M+2]))

    # Build adjacency list (0-based indexing)
    adj = defaultdict(list)
    for i in range(M):
        a = edges[2*i] - 1  # Convert to 0-based
        b = edges[2*i+1] - 1  # Convert to 0-based
        adj[a].append(b)

    # Initialize visited array and min_cycle
    visited = [-1] * N
    min_cycle = float('inf')

    # Iterative DFS stack: (vertex, depth)
    stack = []
    stack.append((0, 0))  # Start from vertex 1 (0-based), depth 0
    visited[0] = 0  # Mark vertex 1 as visited at depth 0

    while stack:
        u, d = stack.pop()
        for v in adj[u]:
            if visited[v] == -1:
                visited[v] = d + 1
                stack.append((v, d + 1))
            elif v == 0:  # Back edge to vertex 1
                cycle_length = d - visited[v] + 1
                if cycle_length < min_cycle:
                    min_cycle = cycle_length

    if min_cycle == float('inf'):
        print(-1)
    else:
        print(min_cycle)

if __name__ == "__main__":
    main()