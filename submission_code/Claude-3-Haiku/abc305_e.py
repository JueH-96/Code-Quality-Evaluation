from collections import deque

def bfs(graph, start, max_distance):
    queue = deque([(start, 0)])
    visited = set()
    while queue:
        vertex, distance = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        if distance <= max_distance:
            yield vertex
        for neighbor in graph[vertex]:
            queue.append((neighbor, distance + 1))

def solve():
    N, M, K = map(int, input().split())
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    guards = []
    for _ in range(K):
        p, h = map(int, input().split())
        guards.append((p, h))

    guarded_vertices = set()
    for p, h in guards:
        for vertex in bfs(graph, p, h):
            guarded_vertices.add(vertex)

    print(len(guarded_vertices))
    print(" ".join(map(str, sorted(guarded_vertices))))

solve()