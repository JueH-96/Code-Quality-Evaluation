from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # adjacency list for the directed graph
        adj = [[] for _ in range(n)]
        # initially, we have roads i -> i+1 for all 0 <= i < n-1
        for i in range(n - 1):
            adj[i].append(i + 1)

        answers = []

        for u, v in queries:
            # add the new road u -> v
            adj[u].append(v)

            # BFS from 0 to find shortest path to n-1
            dist = [-1] * n
            dq = deque([0])
            dist[0] = 0

            while dq:
                node = dq.popleft()
                if node == n - 1:
                    break  # We've reached the target; dist[n-1] is final
                for w in adj[node]:
                    if dist[w] == -1:
                        dist[w] = dist[node] + 1
                        dq.append(w)

            answers.append(dist[n - 1])

        return answers