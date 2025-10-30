class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)

        answer = []
        for query in queries:
            u, v = query
            adj[u].append(v)

            q = [(0, 0)] # (city, distance)
            visited = {0}
            shortest_path = float('inf')

            while q:
                curr_city, dist = q.pop(0)

                if curr_city == n - 1:
                    shortest_path = dist
                    break

                for neighbor in adj[curr_city]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
            answer.append(shortest_path)
        return answer