from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        """
        Calculates the number of times an ant returns to the boundary.

        The ant starts at position 0. For each number in nums, it moves
        according to the value:
        - If nums[i] < 0, moves left by -nums[i] units.
        - If nums[i] > 0, moves right by nums[i] units.
        The ant's position is checked only after each complete move.

        Args:
            nums: A list of non-zero integers representing the ant's movements.

        Returns:
            The number of times the ant returns to the boundary (position 0).
        """
        
        current_position = 0  # Initialize the ant's starting position at the boundary
        boundary_returns_count = 0  # Initialize the counter for boundary returns

        # Iterate through each movement in the nums array
        for move_value in nums:
            # Update the ant's current position based on the move_value
            current_position += move_value
            
            # Check if the ant has returned to the boundary (position 0)
            if current_position == 0:
                boundary_returns_count += 1
                
        # Return the total count of times the ant returned to the boundary
        return boundary_returns_count