class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        times = [0] * n
        visited = [False] * n

        def dfs(node, parent, time):
            visited[node] = True
            times[node] = max(times[node], time)
            for child in graph[node]:
                if child != parent:
                    dfs(child, node, time + 1 if node % 2 == 0 else time + 2)

        for node in range(n):
            if not visited[node]:
                dfs(node, -1, 0)

        return times