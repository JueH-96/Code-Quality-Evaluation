class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            n = len(colors)
            count = 0
            for i in range(n):
                valid = True
                for j in range(size):
                    if colors[(i + j) % n] == colors[(i + j + 1) % n]:
                        valid = False
                        break
                if valid:
                    count += 1
            return count
        
        result = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                result.append(count_alternating_groups(size))
            elif query[0] == 2:
                index, color = query[1], query[2]
                colors[index] = color
        
        return result