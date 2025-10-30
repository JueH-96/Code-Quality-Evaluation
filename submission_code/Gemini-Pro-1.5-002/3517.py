class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)

        ans = []
        for u, v in queries:
            adj[u].append(v)
            q = deque([(0, 0)])
            visited = set()
            shortest_path = float('inf')

            while q:
                curr, dist = q.popleft()
                if curr == n - 1:
                    shortest_path = min(shortest_path, dist)
                    continue

                if curr in visited:
                    continue
                visited.add(curr)

                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        q.append((neighbor, dist + 1))
            
            ans.append(shortest_path)
        return ans