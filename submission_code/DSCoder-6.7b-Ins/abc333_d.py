import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline())
    graph = [[] for _ in range(N+1)]
    in_degree = [0] * (N+1)
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        graph[u].append(v)
        graph[v].append(u)
        in_degree[u] += 1
        in_degree[v] += 1

    queue = deque([i for i in range(2, N+1) if in_degree[i] == 1])
    removed = [False] * (N+1)
    operations = 0

    while queue:
        operations += 1
        for _ in range(len(queue)):
            v = queue.popleft()
            removed[v] = True
            for u in graph[v]:
                if not removed[u]:
                    in_degree[u] -= 1
                    if in_degree[u] == 1:
                        queue.append(u)

    print(operations)

solve()