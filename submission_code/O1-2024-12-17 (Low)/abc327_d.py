def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(lambda x: int(x) - 1, input_data[2:2+M]))
    B = list(map(lambda x: int(x) - 1, input_data[2+M:2+2*M]))

    # Build adjacency list for the graph
    # We have an edge (A[i], B[i]) which implies they must be colored differently
    adj = [[] for _ in range(N)]
    for i in range(M):
        adj[A[i]].append(B[i])
        adj[B[i]].append(A[i])

    # Color array, -1 means uncolored. We want to 2-color the graph.
    color = [-1]*N

    def dfs(start):
        stack = [start]
        color[start] = 0  # start coloring with 0
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    return False
        return True

    # Check each connected component
    for node in range(N):
        if color[node] == -1:
            if not dfs(node):
                print("No")
                return

    print("Yes")

# Do not forget to call main()
if __name__ == "__main__":
    main()