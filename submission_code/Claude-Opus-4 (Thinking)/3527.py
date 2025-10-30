class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count alternating groups
                size = query[1]
                count = 0
                
                # Check each possible starting position
                for start in range(n):
                    # Check if group starting at 'start' with given size is alternating
                    is_alternating = True
                    for i in range(size - 1):
                        # Compare adjacent elements in the group (with wraparound)
                        if colors[(start + i) % n] == colors[(start + i + 1) % n]:
                            is_alternating = False
                            break
                    
                    if is_alternating:
                        count += 1
                
                result.append(count)
            
            else:  # Update color (query[0] == 2)
                colors[query[1]] = query[2]
        
        return result