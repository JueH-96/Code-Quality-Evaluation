def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read N and M
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Read sequences A and B
    A = [int(next(it)) for _ in range(M)]
    B = [int(next(it)) for _ in range(M)]
    
    # Build an undirected graph from constraints
    # constraint: for each i, X[A_i] != X[B_i]
    # which is equivalent to connect node A_i and node B_i with an edge.
    adj = [[] for _ in range(N + 1)]
    
    for a, b in zip(A, B):
        # If the two endpoints are same, then it's impossible because
        # X[A_i] != X[A_i] doesn't make sense.
        if a == b:
            sys.stdout.write("No")
            return
        adj[a].append(b)
        adj[b].append(a)
    
    # To see if assignment is possible we need a bipartite graph.
    color = [-1] * (N + 1)  # -1 means not colored, use colors 0 and 1
    
    # Check each component
    for i in range(1, N + 1):
        if color[i] == -1:
            # BFS for bipartiteness
            queue = deque()
            queue.append(i)
            color[i] = 0
            while queue:
                cur = queue.popleft()
                for nxt in adj[cur]:
                    if color[nxt] == -1:
                        color[nxt] = color[cur] ^ 1
                        queue.append(nxt)
                    elif color[nxt] == color[cur]:
                        sys.stdout.write("No")
                        return
    sys.stdout.write("Yes")

if __name__ == '__main__':
    main()