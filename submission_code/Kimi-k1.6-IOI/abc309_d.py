import sys
from collections import deque

def main():
    n1, n2, m = map(int, sys.stdin.readline().split())
    v = n1 + n2
    adj = [[] for _ in range(v + 1)]
    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS from node 1
    dist1 = [-1] * (v + 1)
    q = deque()
    dist1[1] = 0
    q.append(1)
    while q:
        node = q.popleft()
        for neighbor in adj[node]:
            if dist1[neighbor] == -1:
                dist1[neighbor] = dist1[node] + 1
                q.append(neighbor)
    max1 = max(dist1[1:n1+1])
    
    # BFS from node v
    dist2 = [-1] * (v + 1)
    q = deque()
    dist2[v] = 0
    q.append(v)
    while q:
        node = q.popleft()
        for neighbor in adj[node]:
            if dist2[neighbor] == -1:
                dist2[neighbor] = dist2[node] + 1
                q.append(neighbor)
    max2 = max(dist2[n1+1:v+1])
    
    print(max1 + max2 + 1)

if __name__ == "__main__":
    main()