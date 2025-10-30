from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0

        # Iterate through each tile to consider it as the middle tile of a group
        for i in range(n):
            # Get the color of the current (middle) tile
            current_color = colors[i]

            # Get the color of the tile to the left
            # Using (i - 1 + n) % n handles the circularity for the first tile (index 0)
            left_color = colors[(i - 1 + n) % n]

            # Get the color of the tile to the right
            # Using (i + 1) % n handles the circularity for the last tile (index n-1)
            right_color = colors[(i + 1) % n]

            # Check if the middle tile's color is different from both its left and right neighbors
            if left_color != current_color and current_color != right_color:
                count += 1
        
        return count