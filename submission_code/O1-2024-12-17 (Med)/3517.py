class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque

        # Build the adjacency list with the initial roads 0->1, 1->2, ..., (n-2)->(n-1).
        adjacency = [[] for _ in range(n)]
        for i in range(n - 1):
            adjacency[i].append(i + 1)

        def bfs_shortest_path() -> int:
            # Returns the shortest distance from city 0 to city n-1.
            distance = [-1] * n
            distance[0] = 0
            queue = deque([0])
            while queue:
                node = queue.popleft()
                for nxt in adjacency[node]:
                    if distance[nxt] == -1:  # not visited
                        distance[nxt] = distance[node] + 1
                        # If we've reached city n-1, we can return immediately.
                        if nxt == n - 1:
                            return distance[nxt]
                        queue.append(nxt)
            return -1  # If unreachable (shouldn't happen under given constraints)

        ans = []
        # Process each query by adding the new road and then computing the shortest path.
        for u, v in queries:
            adjacency[u].append(v)
            ans.append(bfs_shortest_path())

        return ans