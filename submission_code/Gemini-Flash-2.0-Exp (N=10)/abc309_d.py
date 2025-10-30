from collections import deque

def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        edges.append(list(map(int, input().split())))

    def bfs(start, end, graph):
        q = deque([(start, 0)])
        visited = {start}
        while q:
            curr, dist = q.popleft()
            if curr == end:
                return dist
            for neighbor in graph.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        return float('inf')

    max_dist = 0
    for u in range(1, n1 + 1):
        for v in range(n1 + 1, n1 + n2 + 1):
            graph = {}
            for a, b in edges:
                graph.setdefault(a, []).append(b)
                graph.setdefault(b, []).append(a)
            
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)

            dist = bfs(1, n1 + n2, graph)
            max_dist = max(max_dist, dist)
    
    print(max_dist)

solve()