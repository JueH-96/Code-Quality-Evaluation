import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N = int(stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v, l = map(int, stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))
    
    path_sum = [0] * (N + 1)
    visited = [False] * (N + 1)
    parent = [0] * (N + 1)
    stack = [(1, 0)]
    while stack:
        u, s = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        path_sum[u] = s
        for v, l in edges[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                stack.append((v, s + l))
    
    nodes = [path_sum[i] for i in range(1, N+1)]
    nodes.sort(reverse=True)
    
    prefix = [0] * (N + 1)
    for i in range(1, N+1):
        prefix[i] = prefix[i-1] + nodes[i-1]
    
    for k in range(1, N+1):
        print(prefix[k] * 2)

if __name__ == '__main__':
    main()