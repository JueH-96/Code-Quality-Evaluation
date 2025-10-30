import sys
from collections import deque

def main():
    n1, n2, m = map(int, sys.stdin.readline().split())
    total_nodes = n1 + n2
    adj = [[] for _ in range(total_nodes + 1)]  # 1-based indexing

    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # Compute distances for group 1 (nodes 1 to n1)
    dist1 = [-1] * (total_nodes + 1)
    dist1[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)
    
    # Compute distances for group 2 (nodes n1+1 to total_nodes)
    s = total_nodes
    dist2 = [-1] * (total_nodes + 1)
    dist2[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)
    
    # Find maximum distances in each group
    max_d1 = max(dist1[1:n1+1])
    max_d2 = max(dist2[n1+1:total_nodes+1])
    
    print(max_d1 + max_d2 + 1)

if __name__ == '__main__':
    main()