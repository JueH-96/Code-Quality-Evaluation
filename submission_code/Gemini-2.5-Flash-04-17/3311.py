from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        """
        Counts the number of times an ant returns to the boundary (position 0).

        Args:
            nums: A list of non-zero integers representing the ant's moves.
                  Positive values mean moving right, negative values mean moving left.

        Returns:
            The number of times the ant returns to the boundary after a move.
        """
        current_position = 0  # Ant starts at the boundary
        boundary_returns = 0  # Counter for returns to boundary

        # Iterate through each move in the input list
        for move in nums:
            # Update the ant's position based on the current move
            current_position += move

            # Check if the ant has returned to the boundary (position 0) after the move
            if current_position == 0:
                boundary_returns += 1

        # Return the total count of boundary returns
        return boundary_returns