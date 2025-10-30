import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Check if node 1 is a leaf
    if len(adj[1]) == 1:
        print(1)
        return
    
    # BFS to find the longest path from node 1
    visited = [False] * (N + 1)
    distance = [0] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    distance[1] = 1
    max_dist = 1
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                distance[v] = distance[u] + 1
                if distance[v] > max_dist:
                    max_dist = distance[v]
                q.append(v)
    
    print(N - max_dist)

if __name__ == '__main__':
    main()