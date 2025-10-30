import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N1 + N2 + 2)]  # 1-based indexing

    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS for first group (1 to N1)
    max_d1 = 0
    d1 = [-1] * (N1 + N2 + 2)
    q = deque()
    q.append(1)
    d1[1] = 0
    max_d1 = 0
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if 1 <= v <= N1 and d1[v] == -1:
                d1[v] = d1[u] + 1
                if d1[v] > max_d1:
                    max_d1 = d1[v]
                q.append(v)
    
    # BFS for second group (N1+1 to N1+N2)
    start = N1 + N2
    max_d2 = 0
    d2 = [-1] * (N1 + N2 + 2)
    q = deque()
    q.append(start)
    d2[start] = 0
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if (N1 + 1 <= v <= N1 + N2) and d2[v] == -1:
                d2[v] = d2[u] + 1
                if d2[v] > max_d2:
                    max_d2 = d2[v]
                q.append(v)
    
    print(max_d1 + max_d2 + 1)

if __name__ == "__main__":
    main()