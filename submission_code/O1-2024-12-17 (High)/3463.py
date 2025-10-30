class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # We treat the array as circular, so use modulo indexing.
        for i in range(n):
            left = colors[i]
            mid = colors[(i + 1) % n]
            right = colors[(i + 2) % n]
            
            if mid != left and mid != right:
                count += 1
        
        return count