class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        result = []
        
        for query in queries:
            if query[0] == 1:
                size = query[1]
                count = 0
                for start in range(n):
                    if self.isAlternatingGroup(colors, start, size, n):
                        count += 1
                result.append(count)
            else:
                idx, color = query[1], query[2]
                colors[idx] = color
        
        return result
    
    def isAlternatingGroup(self, colors, start, size, n):
        for i in range(size - 1):
            current_idx = (start + i) % n
            next_idx = (start + i + 1) % n
            if colors[current_idx] == colors[next_idx]:
                return False
        
        return True