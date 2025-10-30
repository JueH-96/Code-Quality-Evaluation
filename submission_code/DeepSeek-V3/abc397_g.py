import sys
from collections import deque

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    N, M = map(int, input[ptr:ptr+2])
    ptr +=2
    
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        x, y, z = map(int, input[ptr:ptr+3])
        ptr +=3
        adj[x].append((y, z))
        adj[y].append((x, z))
    
    A = [0]*(N+1)
    visited = [False]*(N+1)
    
    for i in range(1, N+1):
        if not visited[i]:
            queue = deque()
            queue.append(i)
            visited[i] = True
            A[i] = 0
            while queue:
                v = queue.popleft()
                for (u, z) in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        A[u] = A[v] ^ z
                        queue.append(u)
                    else:
                        if A[u] != (A[v] ^ z):
                            print(-1)
                            return
    print(' '.join(map(str, A[1:N+1])))

solve()