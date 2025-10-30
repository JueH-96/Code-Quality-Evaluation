from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        """
        Calculates the number of times an ant returns to the boundary (position 0)
        after following a sequence of moves defined by the nums array.

        Args:
            nums: A list of non-zero integers representing the ant's moves.
                  A positive number means move right, a negative number means move left.

        Returns:
            The number of times the ant is exactly at the boundary (position 0)
            after completing a move.
        """
        current_position = 0  # Ant starts at the boundary (position 0)
        boundary_return_count = 0 # Counter for returns to the boundary

        # Iterate through each move in the nums array
        for move in nums:
            # Update the ant's position based on the current move
            current_position += move 
            
            # Check if the ant is exactly at the boundary *after* the move
            if current_position == 0:
                boundary_return_count += 1
                
        return boundary_return_count