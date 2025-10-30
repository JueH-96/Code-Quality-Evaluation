from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        """
        Counts the number of alternating groups of three contiguous tiles in a circular array of colors.

        An alternating group consists of three tiles where the middle tile has a different color
        than its two neighbors (e.g., the pattern is 010 or 101). The array `colors` represents
        the tiles in a circle, meaning the last element is considered adjacent to the first element.

        Args:
            colors: A list of integers where 0 represents red and 1 represents blue.
                    The length of the list `n` satisfies 3 <= n <= 100.
                    Each element `colors[i]` is either 0 or 1.

        Returns:
            An integer representing the total number of alternating groups found in the circular arrangement of tiles.
        """
        n = len(colors)
        alternating_group_count = 0

        # Iterate through each tile index 'i' from 0 to n-1.
        # Each 'i' represents the starting position of a potential group of three contiguous tiles.
        # Since the array is circular, we need to consider groups that wrap around the end.
        for i in range(n):
            # Determine the indices of the three contiguous tiles for the group starting at index i.
            # The modulo operator (%) handles the circular nature of the array, ensuring that
            # indices wrap around correctly (e.g., index n becomes 0, index n+1 becomes 1).
            current_tile_index = i
            next_tile_index = (i + 1) % n
            next_next_tile_index = (i + 2) % n

            # Get the colors of the three tiles in the current group.
            current_tile_color = colors[current_tile_index]
            next_tile_color = colors[next_tile_index]
            next_next_tile_color = colors[next_next_tile_index]

            # Check if the current group of three tiles forms an alternating pattern.
            # An alternating group requires the middle tile's color to be different from
            # the color of its left neighbor AND different from the color of its right neighbor.
            # Condition: color[i] != color[(i+1)%n] AND color[(i+1)%n] != color[(i+2)%n]
            # Given that colors are only 0 or 1, this condition is equivalent to:
            # color[i] == color[(i+2)%n] AND color[i] != color[(i+1)%n]
            if current_tile_color != next_tile_color and next_tile_color != next_next_tile_color:
                # If the condition is met, increment the count of alternating groups.
                alternating_group_count += 1

        # Return the total count of alternating groups found.
        return alternating_group_count