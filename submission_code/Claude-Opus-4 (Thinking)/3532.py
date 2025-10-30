class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import deque
        
        n = len(edges) + 1
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        result = []
        
        for start in range(n):
            # BFS from start node
            marked_time = [-1] * n
            marked_time[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                current_time = marked_time[node]
                
                for neighbor in adj[node]:
                    if marked_time[neighbor] == -1:
                        # Calculate when neighbor can be marked
                        if neighbor % 2 == 1:  # odd
                            marked_time[neighbor] = current_time + 1
                        else:  # even
                            marked_time[neighbor] = current_time + 2
                        queue.append(neighbor)
            
            # The answer is the maximum time
            result.append(max(marked_time))
        
        return result