import sys
from collections import deque

def main():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize adjacency list
    adj = [{} for _ in range(N+1)]
    
    # Read M edges
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        adj[A][B] = (X, Y)
    
    # Initialize coordinates and visited list
    s = [0] * (N+1)
    t = [0] * (N+1)
    visited = [False] * (N+1)
    
    # BFS queue
    queue = deque()
    
    # Start from person 1
    s[1] = 0
    t[1] = 0
    visited[1] = True
    queue.append(1)
    
    # Perform BFS
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                s[v] = s[u] + adj[u][v][0]
                t[v] = t[u] + adj[u][v][1]
                visited[v] = True
                queue.append(v)
    
    # Prepare output
    output = []
    for i in range(1, N+1):
        if visited[i]:
            output.append(f"{s[i]} {t[i]}")
        else:
            output.append("undecidable")
    
    # Print all lines at once
    print('
'.join(output))

if __name__ == '__main__':
    main()