from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1

        # Create the adjacency list for the tree
        tree = defaultdict(list)
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)

        def bfs(start):
            # Queue for BFS with (node, time, mark_type)
            queue = deque([(start, 0, 0)])
            visited = [False] * n
            visited[start] = True
            max_time = 0

            while queue:
                node, time, mark_type = queue.popleft()
                max_time = max(max_time, time)

                for neighbor in tree[node]:
                    if not visited[neighbor]:
                        if neighbor % 2 == 0:
                            if mark_type == 0:
                                queue.append((neighbor, time + 2, 2))
                            else:
                                queue.append((neighbor, time + 1, 1))
                        else:
                            if mark_type == 0:
                                queue.append((neighbor, time + 1, 1))
                            else:
                                queue.append((neighbor, time + 2, 2))
                        visited[neighbor] = True

            return max_time

        result = []
        for i in range(n):
            result.append(bfs(i))

        return result