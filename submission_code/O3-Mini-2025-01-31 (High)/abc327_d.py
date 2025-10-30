def main():
    import sys
    from collections import deque
    data = sys.stdin.read().split()
    if not data:
        return
    # Read N and M
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Read the two sequences A and B
    A = [int(next(it)) for _ in range(M)]
    B = [int(next(it)) for _ in range(M)]
    
    # Build the graph: vertices 0 ... N-1
    # In the graph, every condition X[A_i] != X[B_i] is represented as
    # an edge (A_i-1, B_i-1) which needs to be bipartite.
    graph = [[] for _ in range(N)]
    
    for i in range(M):
        a = A[i] - 1
        b = B[i] - 1
        # A self-loop forces X[a] != X[a], which is impossible
        if a == b:
            sys.stdout.write("No")
            return
        graph[a].append(b)
        graph[b].append(a)
    
    # We'll perform a bipartite check by assigning colors (0 and 1)
    color = [-1] * N  # -1 denotes unvisited
    dq = deque()
    
    for i in range(N):
        if color[i] == -1:
            color[i] = 0
            dq.append(i)
            while dq:
                u = dq.popleft()
                for v in graph[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        dq.append(v)
                    elif color[v] == color[u]:
                        sys.stdout.write("No")
                        return
    sys.stdout.write("Yes")
    
if __name__ == '__main__':
    main()