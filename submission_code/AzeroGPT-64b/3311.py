class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        """
        Counts the number of times the ant returns to the boundary based on the given array of movements.
        
        :param nums: List[int] - a list of integers representing the ant's movements.
        :return: int - the number of times the ant returns to the boundary.
        """
        count = 0  # Initialize the count of boundary returns.
        position = 0  # Initialize the ant's position.
        
        # Iterate through the movements.
        for movement in nums:
            position += movement  # Update the ant's position.
            
            # Check if the ant is on the boundary after moving.
            if position == 0:
                count += 1  # Increment the count if the ant is on the boundary.
        
        return count  # Return the count of boundary returns.