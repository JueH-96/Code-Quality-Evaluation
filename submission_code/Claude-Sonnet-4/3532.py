class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict, deque
        
        n = len(edges) + 1
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def simulate(start):
            # marked[node] = time when node was marked, -1 if not marked
            marked = [-1] * n
            marked[start] = 0
            
            time = 1
            while True:
                newly_marked = []
                
                for node in range(n):
                    if marked[node] != -1:  # already marked
                        continue
                    
                    # Check if this node can be marked at current time
                    can_mark = False
                    
                    if node % 2 == 1:  # odd node
                        # needs neighbor marked at time-1
                        for neighbor in graph[node]:
                            if marked[neighbor] == time - 1:
                                can_mark = True
                                break
                    else:  # even node
                        # needs neighbor marked at time-2
                        if time >= 2:
                            for neighbor in graph[node]:
                                if marked[neighbor] == time - 2:
                                    can_mark = True
                                    break
                    
                    if can_mark:
                        newly_marked.append(node)
                
                if not newly_marked:
                    break
                
                # Mark all newly marked nodes
                for node in newly_marked:
                    marked[node] = time
                
                time += 1
            
            # Return the maximum time (when last node was marked)
            return max(marked)
        
        result = []
        for i in range(n):
            result.append(simulate(i))
        
        return result