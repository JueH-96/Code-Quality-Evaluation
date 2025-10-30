import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    reverse_edges = []
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        if b == 1:
            reverse_edges.append(a)
    
    distance = [-1] * (N + 1)
    q = deque([1])
    distance[1] = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                q.append(v)
    
    min_length = float('inf')
    for u in reverse_edges:
        if distance[u] != -1:
            min_length = min(min_length, distance[u] + 1)
    
    print(-1 if min_length == float('inf') else min_length)

if __name__ == "__main__":
    main()