import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    total = N1 + N2
    adj = [[] for _ in range(total + 1)]
    
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # Compute distances from node 1
    dist1 = [-1] * (total + 1)
    q = deque()
    q.append(1)
    dist1[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)
    
    # Find maximum distance in the first component
    max1 = max(dist1[1:N1+1])
    
    # Compute distances from node total (N1 + N2)
    dist2 = [-1] * (total + 1)
    q.append(total)
    dist2[total] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)
    
    # Find maximum distance in the second component
    max2 = max(dist2[N1+1 : total+1])
    
    print(max1 + max2 + 1)

if __name__ == "__main__":
    main()