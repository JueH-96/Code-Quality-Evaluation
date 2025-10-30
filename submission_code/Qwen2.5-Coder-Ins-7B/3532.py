class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node, parent, time):
            times[node] = time
            max_time = time
            for neighbor in graph[node]:
                if neighbor != parent:
                    if node % 2 == 0:
                        max_time = max(max_time, dfs(neighbor, node, time + 2))
                    else:
                        max_time = max(max_time, dfs(neighbor, node, time + 1))
            return max_time
        
        times = [0] * n
        dfs(0, -1, 0)
        return times