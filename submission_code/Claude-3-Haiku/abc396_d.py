from collections import deque

def bfs(graph, start, end):
    queue = deque([(start, 0)])
    visited = set([start])
    min_xor = float('inf')

    while queue:
        node, xor_val = queue.popleft()
        if node == end:
            min_xor = min(min_xor, xor_val)
            continue

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, xor_val ^ weight))

    return min_xor

def solve():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N+1)]

    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    return bfs(graph, 1, N)

print(solve())