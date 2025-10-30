import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    total_nodes = N1 + N2
    adj = [[] for _ in range(total_nodes + 1)]
    
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # Compute distances for the first component starting from node 1
    dist1 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(1)
    dist1[1] = 0
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)
    
    max_dist1 = max(dist1[i] for i in range(1, N1 + 1))
    
    # Compute distances for the second component starting from node (N1 + N2)
    dist2 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(N1 + N2)
    dist2[N1 + N2] = 0
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)
    
    max_dist2 = max(dist2[i] for i in range(N1 + 1, N1 + N2 + 1))
    
    print(max_dist1 + max_dist2 + 1)

if __name__ == '__main__':
    main()