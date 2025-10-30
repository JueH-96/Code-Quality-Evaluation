from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        answer = []
        edges = set()
        # Add initial edges
        for i in range(n - 1):
            edges.add((i, i + 1))

        for query in queries:
            u, v = query
            edges.add((u, v))

            # Perform BFS
            adj_list = [[] for _ in range(n)]
            for u_edge, v_edge in edges:
                adj_list[u_edge].append(v_edge)

            dist = [-1] * n
            dist[0] = 0
            queue = deque([0])

            while queue:
                curr = queue.popleft()
                for neighbor in adj_list[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        queue.append(neighbor)

            answer.append(dist[n - 1] if dist[n - 1] != -1 else float('inf'))

        return [ans if ans != float('inf') else -1 for ans in answer]