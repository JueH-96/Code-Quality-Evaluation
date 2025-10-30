import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    edges = [[] for _ in range(n + 1)]
    degree = [0] * (n + 1)
    for _ in range(n - 1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
        degree[u] += 1
        degree[v] += 1

    queue = deque()
    for i in range(1, n + 1):
        if degree[i] == 1:
            queue.append(i)

    result = []
    visited = [False] * (n + 1)

    while queue:
        u = queue.popleft()
        for v in edges[u]:
            if not visited[v]:
                visited[v] = True
                result.append(degree[v])
                degree[v] -= 1
                if degree[v] == 1:
                    queue.append(v)

    result.sort()
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()