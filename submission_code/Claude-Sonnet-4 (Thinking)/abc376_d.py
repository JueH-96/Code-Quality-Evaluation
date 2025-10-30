from collections import deque

def solve():
    n, m = map(int, input().split())
    
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)
    
    dist = [-1] * (n + 1)
    dist[1] = 0
    queue = deque([1])
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v == 1:
                return dist[u] + 1
            elif dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    
    return -1

print(solve())