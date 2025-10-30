class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            if size > len(colors):
                return 0
            
            count = 0
            n = len(colors)
            
            for start in range(n):
                # Check if we have an alternating group starting at 'start' with length 'size'
                is_alternating = True
                for i in range(1, size):
                    curr_idx = (start + i) % n
                    prev_idx = (start + i - 1) % n
                    if colors[curr_idx] == colors[prev_idx]:
                        is_alternating = False
                        break
                
                if is_alternating:
                    count += 1
            
            return count
        
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count alternating groups
                size = query[1]
                result.append(count_alternating_groups(size))
            else:  # Change color
                index = query[1]
                new_color = query[2]
                colors[index] = new_color
        
        return result