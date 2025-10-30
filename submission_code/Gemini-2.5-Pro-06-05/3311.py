from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        """
        Calculates the number of times an ant returns to its starting boundary.

        The ant's position is tracked as a running sum of the movements given in `nums`.
        The ant starts at position 0. Each number in `nums` represents a move.
        We count how many times the position becomes 0 *after* a move is completed.

        Args:
            nums: A list of non-zero integers representing the ant's movements.

        Returns:
            The number of times the ant is at the boundary (position 0) after a move.
        """
        
        # The ant starts at the boundary, which we can represent as position 0.
        position = 0
        
        # This counter will store the number of times the ant returns to the boundary.
        count = 0
        
        # Iterate through each move in the nums array.
        for move in nums:
            # Update the ant's position. A positive move increases the position (right),
            # and a negative move decreases it (left).
            position += move
            
            # After each move, check if the ant is back at the boundary.
            if position == 0:
                count += 1
                
        return count