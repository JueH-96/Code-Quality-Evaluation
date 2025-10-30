class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check each position as the middle tile
        for i in range(n):
            # Get the three tiles: left, middle, right
            # Use modulo to handle circular nature
            left = colors[(i - 1) % n]
            middle = colors[i]
            right = colors[(i + 1) % n]
            
            # Check if it's an alternating group
            # Middle tile must be different from both neighbors
            if left != middle and middle != right:
                count += 1
        
        return count