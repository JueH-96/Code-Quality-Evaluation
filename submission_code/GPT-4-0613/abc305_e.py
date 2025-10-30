from collections import deque
from sys import stdin, stdout

def bfs(graph, start, stamina):
    visited = [False] * (len(graph) + 1)
    visited[start] = True
    queue = deque([(start, stamina)])
    guarded = [start]
    while queue:
        vertex, stamina = queue.popleft()
        if stamina > 0:
            for neighbour in graph[vertex]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    queue.append((neighbour, stamina - 1))
                    guarded.append(neighbour)
    return guarded

def main():
    N, M, K = map(int, stdin.readline().split())
    graph = {i: [] for i in range(1, N + 1)}
    for _ in range(M):
        a, b = map(int, stdin.readline().split())
        graph[a].append(b)
        graph[b].append(a)
    guards = []
    for _ in range(K):
        p, h = map(int, stdin.readline().split())
        guards.append((p, h))
    guarded_vertices = set()
    for guard in guards:
        guarded_vertices.update(bfs(graph, *guard))
    guarded_vertices = sorted(list(guarded_vertices))
    stdout.write(str(len(guarded_vertices)) + '
')
    stdout.write(' '.join(map(str, guarded_vertices)) + '
')

main()