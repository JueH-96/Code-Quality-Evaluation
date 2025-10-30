from collections import defaultdict
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def dfs(node, parent, start_time):
            max_time = start_time
            for neighbor in graph[node]:
                if neighbor != parent:
                    if node % 2 == 0:  # even node
                        child_time = dfs(neighbor, node, start_time + 2)
                    else:  # odd node
                        child_time = dfs(neighbor, node, start_time + 1)
                    max_time = max(max_time, child_time)
            return max_time

        times = []
        for i in range(n):
            times.append(dfs(i, -1, 0))

        return times