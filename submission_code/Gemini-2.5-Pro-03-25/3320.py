from typing import List

class Solution:
    """
    Calculates the maximum number of operations that can be performed on an array
    such that all operations have the same score. An operation consists of deleting
    the first two elements and its score is their sum.
    """
    def maxOperations(self, nums: List[int]) -> int:
        """
        Finds the maximum number of operations with a consistent score.

        Args:
            nums: A list of integers. Constraints: 2 <= len(nums) <= 100, 1 <= nums[i] <= 1000.

        Returns:
            The maximum number of operations possible such that all performed operations
            have the same score, which is determined by the sum of the first two elements.
        """

        # Constraints guarantee nums.length >= 2, so the first operation is always possible
        # and we can safely access nums[0] and nums[1].
        n = len(nums)
        
        # Calculate the score of the first operation. This score must be maintained
        # for all subsequent operations.
        target_score = nums[0] + nums[1]
        
        # Initialize the count of operations. The first operation is counted here.
        operations_count = 1
        
        # Initialize the index `i` to point to the start of the next potential pair.
        # After the first operation (conceptually removing nums[0] and nums[1]), the 
        # remaining elements effectively start at index 2.
        i = 2
        
        # Iterate through the rest of the array in steps of 2, as long as there are
        # enough elements to form a pair (at least two elements starting from index i).
        # The condition `i + 1 < n` checks if indices `i` and `i+1` are within the bounds
        # of the array.
        while i + 1 < n:
            # Calculate the score of the current pair (nums[i], nums[i+1]).
            current_score = nums[i] + nums[i+1]
            
            # Check if the score of the current pair matches the target score
            # determined by the first operation.
            if current_score == target_score:
                # If the scores match, it means we can perform this operation.
                # Increment the count of successful operations.
                operations_count += 1
                # Move the index forward by 2 to consider the next pair.
                i += 2
            else:
                # If the scores do not match, we cannot continue performing operations
                # with the same score according to the problem rules. Stop the loop.
                break
                
        # Return the total count of operations performed with the consistent score.
        return operations_count