class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        ans = []
        graph = {i: [i + 1] if i < n - 1 else [] for i in range(n)}
        
        def bfs(graph):
            queue = [(0, 0)]
            visited = {0}
            while queue:
                curr, dist = queue.pop(0)
                if curr == n - 1:
                    return dist
                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return float('inf')

        for u, v in queries:
            graph[u].append(v)
            ans.append(bfs(graph))
        return ans