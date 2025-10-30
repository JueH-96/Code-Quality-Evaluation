class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        n = len(edges) + 1
        graph = defaultdict(list)
        
        # Build the graph
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def simulate(start):
            marked = [False] * n
            marked_time = [-1] * n
            marked[start] = True
            marked_time[start] = 0
            
            time = 0
            while not all(marked):
                time += 1
                new_marks = []
                
                for node in range(n):
                    if marked[node]:
                        continue
                    
                    # Check if this node can be marked at current time
                    can_mark = False
                    if node % 2 == 1:  # odd node
                        # needs adjacent marked at time-1
                        for neighbor in graph[node]:
                            if marked[neighbor] and marked_time[neighbor] == time - 1:
                                can_mark = True
                                break
                    else:  # even node
                        # needs adjacent marked at time-2
                        if time >= 2:
                            for neighbor in graph[node]:
                                if marked[neighbor] and marked_time[neighbor] == time - 2:
                                    can_mark = True
                                    break
                    
                    if can_mark:
                        new_marks.append(node)
                
                # Mark all new nodes
                for node in new_marks:
                    marked[node] = True
                    marked_time[node] = time
            
            return time
        
        result = []
        for i in range(n):
            result.append(simulate(i))
        
        return result