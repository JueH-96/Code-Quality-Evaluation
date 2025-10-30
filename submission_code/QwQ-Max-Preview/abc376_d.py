import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    back_edges = []
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        if b == 1:
            back_edges.append(a)
    
    # BFS from 1
    distance = [-1] * (N + 1)
    q = deque()
    distance[1] = 0
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                q.append(v)
    
    # Check all back edges
    min_cycle = float('inf')
    for a in back_edges:
        if distance[a] != -1:
            current = distance[a] + 1
            if current < min_cycle:
                min_cycle = current
    
    if min_cycle == float('inf'):
        print(-1)
    else:
        print(min_cycle)

if __name__ == "__main__":
    main()