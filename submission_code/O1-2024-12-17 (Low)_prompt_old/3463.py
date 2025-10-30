class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Loop over each tile index (considering the circular arrangement)
        for i in range(n):
            # Identify the 3 contiguous tiles in the circle
            i1 = (i + 1) % n
            i2 = (i + 2) % n
            
            # Check if the group of tiles (i, i1, i2) alternates in color:
            #   - The middle tile is different from both sides,
            #   - The left and right tiles have the same color.
            if colors[i] == colors[i2] and colors[i] != colors[i1]:
                count += 1
        
        return count