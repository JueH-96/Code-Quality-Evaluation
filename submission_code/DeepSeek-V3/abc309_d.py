import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N1 + N2 + 1)]
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    
    # BFS for the first component (1 to N1)
    dist1 = [-1] * (N1 + N2 + 1)
    q = deque()
    q.append(1)
    dist1[1] = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)
    
    # BFS for the second component (N1+1 to N1+N2)
    dist2 = [-1] * (N1 + N2 + 1)
    q.append(N1 + N2)
    dist2[N1 + N2] = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)
    
    # Find the maximum d
    max_d = 0
    for u in range(1, N1 + 1):
        if dist1[u] != -1:
            for v in range(N1 + 1, N1 + N2 + 1):
                if dist2[v] != -1:
                    current_d = dist1[u] + 1 + dist2[v]
                    if current_d > max_d:
                        max_d = current_d
    print(max_d)

if __name__ == "__main__":
    main()