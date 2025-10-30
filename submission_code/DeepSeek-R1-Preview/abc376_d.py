import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    to_1_edges = []
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        if b == 1:
            to_1_edges.append(a)
    
    distance = [-1] * (N + 1)
    q = deque()
    q.append(1)
    distance[1] = 0
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                q.append(v)
    
    min_cycle = float('inf')
    for u in to_1_edges:
        if distance[u] != -1:
            min_cycle = min(min_cycle, distance[u] + 1)
    
    if min_cycle == float('inf'):
        print(-1)
    else:
        print(min_cycle)

if __name__ == '__main__':
    main()