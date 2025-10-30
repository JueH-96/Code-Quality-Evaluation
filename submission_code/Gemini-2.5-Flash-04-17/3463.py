from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        """
        Calculates the number of alternating groups in a circular array of colors.

        An alternating group is defined as three contiguous tiles in the circle
        where the middle tile has a different color from its left and right tiles.

        Args:
            colors: A list of integers representing the colors of tiles (0 for red, 1 for blue).

        Returns:
            The number of alternating groups.
        """
        n = len(colors)
        count = 0

        # Iterate through each tile in the circle.
        # We consider each tile 'i' as the potential middle tile of a group.
        for i in range(n):
            # Calculate the index of the left neighbor of tile 'i'.
            # We use the modulo operator (%) to handle the circular nature.
            # (i - 1 + n) % n correctly wraps around from index 0 to n-1.
            left_idx = (i - 1 + n) % n

            # Calculate the index of the right neighbor of tile 'i'.
            # (i + 1) % n correctly wraps around from index n-1 to 0.
            right_idx = (i + 1) % n

            # Check if the current tile colors[i] has a different color
            # than its left neighbor colors[left_idx] AND
            # if colors[i] has a different color than its right neighbor colors[right_idx].
            # This condition defines an alternating group centered at index i:
            # color[left_idx], color[i], color[right_idx] where color[i] != color[left_idx]
            # and color[i] != color[right_idx].
            if colors[i] != colors[left_idx] and colors[i] != colors[right_idx]:
                # If it forms an alternating group, increment the counter.
                count += 1

        # Return the total number of alternating groups found.
        return count