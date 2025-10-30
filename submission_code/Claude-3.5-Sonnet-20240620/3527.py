class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        
        def count_alternating_groups(size):
            count = 0
            for i in range(n):
                is_alternating = True
                for j in range(size):
                    if colors[(i + j) % n] == colors[(i + j + 1) % n]:
                        is_alternating = False
                        break
                if is_alternating:
                    count += 1
            return count
        
        def update_color(index, color):
            colors[index] = color
        
        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count_alternating_groups(query[1]))
            else:
                update_color(query[1], query[2])
        
        return result