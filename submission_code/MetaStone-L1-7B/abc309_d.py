import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    total_nodes = N1 + N2
    adj = [[] for _ in range(total_nodes + 2)]
    
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # Compute d1 for component 1 (1..N1)
    d1 = [-1] * (N1 + 1)
    d1[1] = 0
    q = deque()
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d1[v] == -1:
                d1[v] = d1[u] + 1
                q.append(v)
    
    max_d1 = max(d1[1:N1+1])
    
    # Compute d2 for component 2 (N1+1..N1+N2)
    target = N1 + N2
    d2 = [-1] * (target + 1)
    d2[target] = 0
    q = deque()
    q.append(target)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d2[v] == -1:
                d2[v] = d2[u] + 1
                q.append(v)
    
    max_d2 = max(d2[N1+1:target+1])
    
    print(max_d1 + max_d2 + 1)

if __name__ == '__main__':
    main()