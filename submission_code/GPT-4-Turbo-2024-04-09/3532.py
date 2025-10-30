class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict, deque
        
        def bfs(start, is_odd):
            queue = deque([(start, 0)])  # (current node, current time)
            marked_time = [-1] * n
            marked_time[start] = 0
            
            while queue:
                node, time = queue.popleft()
                
                for neighbor in graph[node]:
                    if marked_time[neighbor] == -1:
                        if (is_odd and neighbor % 2 == 1) or (not is_odd and neighbor % 2 == 0):
                            delay = 1
                        else:
                            delay = 2
                        marked_time[neighbor] = time + delay
                        queue.append((neighbor, time + delay))
            
            return max(marked_time)
        
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        result = []
        for i in range(n):
            result.append(bfs(i, i % 2 == 1))
        
        return result