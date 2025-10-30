class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        for i in range(n):
            # Get the three consecutive tiles in circular fashion
            left = colors[i]
            middle = colors[(i + 1) % n]
            right = colors[(i + 2) % n]
            
            # Check if it's an alternating group
            if left != middle and middle != right:
                count += 1
        
        return count