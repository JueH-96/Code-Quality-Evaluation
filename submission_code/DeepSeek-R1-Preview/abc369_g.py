import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))
    
    depth = [0] * (N + 1)
    visited = [False] * (N + 1)
    stack = [(1, 0)]
    while stack:
        u, d = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        depth[u] = d
        for v, l in edges[u]:
            if not visited[v]:
                stack.append((v, d + l))
    
    depths = sorted([depth[i] for i in range(1, N+1)], reverse=True)
    total = 0
    for k in range(1, N+1):
        if k-1 < len(depths):
            total += depths[k-1]
        print(2 * total)
    
if __name__ == '__main__':
    main()