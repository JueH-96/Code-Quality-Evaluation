import collections

def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    def bfs(start, graph):
        queue = collections.deque([start])
        visited = {start}
        distances = {start: 0}
        while queue:
            curr = queue.popleft()
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances

    max_dist = 0
    for u in range(1, n1 + 1):
        for v in range(n1 + 1, n1 + n2 + 1):
            graph = collections.defaultdict(list)
            for a, b in edges:
                graph[a].append(b)
                graph[b].append(a)
            graph[u].append(v)
            graph[v].append(u)
            
            distances = bfs(1, graph)
            if n1 + n2 in distances:
                max_dist = max(max_dist, distances[n1 + n2])

    print(max_dist)

solve()