import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    info = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    info.sort(key=lambda x: x[0])

    x = [0]*N
    y = [0]*N
    visited = [False]*N
    visited[0] = True

    graph = defaultdict(list)
    for a, b, dx, dy in info:
        a -= 1
        b -= 1
        graph[a].append((b, dx, dy))
        graph[b].append((a, -dx, -dy))

    def dfs(v):
        for nv, dx, dy in graph[v]:
            if visited[nv]:
                if x[nv] != x[v] + dx or y[nv] != y[v] + dy:
                    return False
            else:
                x[nv] = x[v] + dx
                y[nv] = y[v] + dy
                visited[nv] = True
                if not dfs(nv):
                    return False
        return True

    for v in range(N):
        if not visited[v] and not dfs(v):
            print('
'.join(['undecidable']*N))
            return

    for i in range(N):
        print(x[i], y[i])

if __name__ == "__main__":
    main()