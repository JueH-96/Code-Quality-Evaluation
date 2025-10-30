class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            count = 0
            n = len(colors)
            for start in range(n):
                is_alternating = True
                for i in range(size - 1):
                    curr_idx = (start + i) % n
                    next_idx = (start + i + 1) % n
                    if colors[curr_idx] == colors[next_idx]:
                        is_alternating = False
                        break
                if is_alternating:
                    count += 1
            return count
        
        result = []
        for query in queries:
            if query[0] == 1:  # Count query
                size = query[1]
                result.append(count_alternating_groups(size))
            else:  # Change query
                index, color = query[1], query[2]
                colors[index] = color
        
        return result