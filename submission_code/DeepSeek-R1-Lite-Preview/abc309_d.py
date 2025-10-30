import sys
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N1, N2, M = map(int, sys.stdin.readline().split())
    N = N1 + N2
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS from 1
    dist1 = [-1] * (N+1)
    q = deque([1])
    dist1[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)
    
    # BFS from N1+N2
    dist2 = [-1] * (N+1)
    q = deque([N1+N2])
    dist2[N1+N2] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)
    
    # Find the maximum d
    max_d = 0
    for u in range(1, N1+1):
        if dist1[u] == -1:
            continue
        for v in range(N1+1, N1+N2+1):
            if dist2[v] == -1:
                continue
            d = dist1[u] + 1 + dist2[v]
            if d > max_d:
                max_d = d
    print(max_d)

if __name__ == '__main__':
    main()