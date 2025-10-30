class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check each position as the middle element of a potential alternating group
        for i in range(n):
            # Get the three consecutive elements (handling circular nature)
            left = colors[(i - 1) % n]
            middle = colors[i]
            right = colors[(i + 1) % n]
            
            # Check if middle is different from both left and right
            if middle != left and middle != right:
                count += 1
        
        return count