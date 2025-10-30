def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    B = list(map(int, data[2+M:2+2*M]))
    
    # If any pair (A_i, B_i) has A_i == B_i, it's impossible to satisfy X[A_i] != X[B_i].
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        adj[A[i]].append(B[i])
        adj[B[i]].append(A[i])
    
    # color[i] will hold the color of node i (0 or 1). -1 means uncolored.
    color = [-1] * (N + 1)
    
    # Use BFS to color the graph, checking for bipartiteness.
    for start in range(1, N + 1):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        print("No")
                        return
    
    print("Yes")

# Do not forget to call main().
if __name__ == "__main__":
    main()