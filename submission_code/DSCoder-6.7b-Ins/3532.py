from collections import defaultdict, deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        indegree = [0]*n
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            indegree[u] += 1
            indegree[v] += 1
        
        queue = deque([i for i in range(n) if indegree[i] == 1])
        times = [0]*n
        while queue:
            new_queue = deque()
            for u in queue:
                for v in graph[u]:
                    indegree[v] -= 1
                    if indegree[v] == 1:
                        new_queue.append(v)
                        times[v] = times[u] + 1 if v % 2 == u % 2 else times[u] + 2
            queue = new_queue
        return times