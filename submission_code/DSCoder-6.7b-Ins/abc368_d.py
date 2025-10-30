import sys
from collections import defaultdict, deque

def read_input():
    N, K = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(N-1)]
    vertices = list(map(int, sys.stdin.readline().split()))
    return N, K, edges, vertices

def solve(N, K, edges, vertices):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    queue = deque([vertices[0]])
    visited = [False] * (N+1)
    visited[vertices[0]] = True
    count = 1

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if not visited[v] and v in vertices:
                queue.append(v)
                visited[v] = True
                count += 1

    return N - count

N, K, edges, vertices = read_input()
print(solve(N, K, edges, vertices))