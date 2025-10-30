import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)

    def bfs(start, tree):
        visited = [False] * (N + 1)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        max_dist = 0
        far_node = start
        while q:
            u, d = q.popleft()
            if d > max_dist:
                max_dist = d
                far_node = u
            for v in tree[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append((v, d+1))
        return far_node, max_dist

    total = 0

    def dfs(u, parent):
        nonlocal total
        if len(edges[u]) == 1 and u != 1 and u != N:
            return
        max_dist = 0
        far_node = u
        for v in edges[u]:
            if v != parent:
                child, dist = dfs(v, u)
                if dist > max_dist:
                    max_dist = dist
                    far_node = v
        total += max_dist
        return far_node, max_dist

    u, _ = bfs(1, edges)
    v, d = bfs(u, edges)
    total += d
    dfs(u, -1)
    print(total)

if __name__ == '__main__':
    main()