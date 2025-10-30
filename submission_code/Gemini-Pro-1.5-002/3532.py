class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def solve(start_node):
            time = [-1] * n
            time[start_node] = 0
            max_time = 0
            q = [(start_node, 0)]
            while q:
                u, t = q.pop(0)
                max_time = max(max_time, t)
                for v in graph[u]:
                    if time[v] == -1:
                        if v % 2 == 1:
                            time[v] = t + 1
                            q.append((v, t + 1))
                        else:
                            time[v] = t + 2
                            q.append((v, t + 2))
            return max_time
        
        times = []
        for i in range(n):
            times.append(solve(i))
        return times