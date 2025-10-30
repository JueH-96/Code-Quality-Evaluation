class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        def count_alternating_groups(size):
            if size > len(colors):
                return 0
            count = 0
            valid = True
            for i in range(size):
                if i > 0 and colors[i] == colors[i - 1]:
                    valid = False
            if valid:
                count += 1
            
            for i in range(size, len(colors) + size - 1):
                index = i % len(colors)
                prev_index = (i - size) % len(colors)
                if colors[prev_index] == colors[(prev_index + 1) % len(colors)]:
                    valid = True
                if colors[(index - 1) % len(colors)] == colors[index]:
                    valid = False
                if valid:
                    count += 1
            
            return count
        
        results = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                results.append(count_alternating_groups(size))
            elif query[0] == 2:
                index, color = query[1], query[2]
                colors[index] = color
        
        return results