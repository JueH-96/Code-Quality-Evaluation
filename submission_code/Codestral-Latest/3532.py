class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import deque, defaultdict

        n = len(edges) + 1
        graph = defaultdict(list)

        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def bfs(start):
            queue = deque([(start, 0)])
            visited = [-1] * n
            visited[start] = 0

            while queue:
                node, time = queue.popleft()
                for neighbor in graph[node]:
                    if visited[neighbor] == -1:
                        if neighbor % 2 == 1:
                            visited[neighbor] = time + 1
                        else:
                            visited[neighbor] = time + 2
                        queue.append((neighbor, visited[neighbor]))

            return max(visited)

        times = []
        for i in range(n):
            times.append(bfs(i))

        return times