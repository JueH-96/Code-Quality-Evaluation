class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        result = []
        
        # For each starting node
        for start in range(n):
            # BFS to find time to mark all nodes
            times = [-1] * n
            times[start] = 0
            queue = deque([start])
            max_time = 0
            
            while queue:
                curr = queue.popleft()
                curr_time = times[curr]
                
                for neighbor in graph[curr]:
                    if times[neighbor] == -1:
                        # Calculate time based on whether neighbor is odd or even
                        if neighbor % 2 == 1:  # odd node
                            times[neighbor] = curr_time + 1
                        else:  # even node
                            times[neighbor] = curr_time + 2
                        
                        max_time = max(max_time, times[neighbor])
                        queue.append(neighbor)
            
            result.append(max_time)
        
        return result