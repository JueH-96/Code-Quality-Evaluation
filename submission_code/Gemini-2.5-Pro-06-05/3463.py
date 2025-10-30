from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # We check every possible group of 3 contiguous tiles.
        # A group can be identified by its starting index 'i'.
        # The loop iterates through all n possible starting indices.
        for i in range(n):
            # The indices of the 3 tiles in the group starting at 'i' are:
            # i, (i + 1) % n, and (i + 2) % n.
            # The modulo operator (%) correctly handles the circular array.
            
            # An alternating group has a pattern like [color_A, color_B, color_A].
            # This means the first and third tiles have the same color, and the 
            # middle tile has a different color. This matches the problem's 
            # definition for binary-colored tiles: "the middle tile has a different
            # color from its left and right tiles".
            
            first_tile_color = colors[i]
            second_tile_color = colors[(i + 1) % n]
            third_tile_color = colors[(i + 2) % n]
            
            if first_tile_color == third_tile_color and first_tile_color != second_tile_color:
                count += 1
        
        return count