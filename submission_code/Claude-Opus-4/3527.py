class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count query
                size = query[1]
                count = 0
                
                # Check each possible starting position
                for start in range(n):
                    is_alternating = True
                    
                    # Check if we have an alternating group of the required size
                    for i in range(size - 1):
                        curr_idx = (start + i) % n
                        next_idx = (start + i + 1) % n
                        
                        if colors[curr_idx] == colors[next_idx]:
                            is_alternating = False
                            break
                    
                    if is_alternating:
                        count += 1
                
                result.append(count)
                
            else:  # Update query
                index = query[1]
                color = query[2]
                colors[index] = color
        
        return result