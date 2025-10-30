import sys
import sys
import sys
def solve():
    import sys
    import sys
    from collections import deque
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        A = int(input[idx]); idx +=1
        B = int(input[idx]); idx +=1
        X = int(input[idx]); idx +=1
        Y = int(input[idx]); idx +=1
        adj[A].append((B, X, Y))
        adj[B].append((A, -X, -Y))
    x = [None]*(N+1)
    y = [None]*(N+1)
    x[1] = 0
    y[1] = 0
    q = deque()
    q.append(1)
    while q:
        u = q.popleft()
        for v, dx, dy in adj[u]:
            if x[v] is None:
                x[v] = x[u] + dx
                y[v] = y[u] + dy
                q.append(v)
            else:
                if x[v] != x[u] + dx or y[v] != y[u] + dy:
                    # Consistent input, so no action needed
                    pass
    output = []
    for i in range(1, N+1):
        if x[i] is not None:
            output.append(f"{x[i]} {y[i]}")
        else:
            output.append("undecidable")
    print("
".join(output))