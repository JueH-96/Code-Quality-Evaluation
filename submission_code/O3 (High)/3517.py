from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # adjacency list – start with the initial chain 0→1→2→…→n-1
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)

        ans = []

        for u, v in queries:
            # add the new road
            adj[u].append(v)

            # BFS from city 0 to city n-1 (all edges weight 1)
            dist = [-1] * n
            q = deque([0])
            dist[0] = 0

            while q:
                cur = q.popleft()
                if cur == n - 1:          # reached the target early
                    break
                for nxt in adj[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)

            ans.append(dist[n - 1])

        return ans