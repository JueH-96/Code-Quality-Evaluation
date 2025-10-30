def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = [int(x) - 1 for x in data[2:2 + M]]
    B = [int(x) - 1 for x in data[2 + M:2 + 2 * M]]

    # Build adjacency list
    adj = [[] for _ in range(N)]
    for i in range(M):
        a, b = A[i], B[i]
        # If A[i] == B[i], it's impossible to have different values for X_a and X_b
        if a == b:
            print("No")
            return
        adj[a].append(b)
        adj[b].append(a)

    # Color array, -1 means uncolored, then we use 0 and 1 for bipartite check
    color = [-1] * N

    # Check bipartite via BFS on each connected component
    for start in range(N):
        if color[start] == -1:  # unvisited
            color[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        print("No")
                        return

    print("Yes")

# Do not forget to call main
if __name__ == "__main__":
    main()