import math # This import is not strictly needed for this solution, but included in case of future modifications or standard practice. It can be removed.
from typing import List # Import List for type hinting

class Solution:
    """
    Solves the problem of calculating the matrix sum based on repeatedly removing 
    the largest element from each row and adding the maximum of the removed elements to the score.
    The approach uses sorting each row first, which simplifies the process.
    """
    def matrixSum(self, nums: List[List[int]]) -> int:
        """
        Calculates the final score according to the described process using an efficient sorting-based approach.

        The core idea is that if we sort each row, then the process described in the problem 
        is equivalent to iterating through the columns of the sorted matrix (from right to left, conceptually)
        and finding the maximum element in each column, then summing these maximums. Iterating 
        columns from left to right and summing column maximums achieves the same result.

        Args:
            nums: A 0-indexed 2D integer array.

        Returns:
            The final score accumulated through the described operations.
        """

        # Get the dimensions of the matrix. 
        # Constraints state 1 <= nums.length and 1 <= nums[i].length, so the matrix is non-empty.
        num_rows = len(nums)
        num_cols = len(nums[0])

        # Step 1: Sort each row of the matrix in ascending order.
        # This is the key step that simplifies the problem. After sorting, 
        # the largest element of each original row `i` is now at `nums[i][num_cols - 1]`,
        # the second largest at `nums[i][num_cols - 2]`, and so on.
        for i in range(num_rows):
            nums[i].sort() # Sorts the list in-place.

        # Initialize the total score
        total_score = 0

        # Step 2: Iterate through the columns of the row-sorted matrix.
        # For each column index `j`, find the maximum element across all rows in that column.
        # This maximum value corresponds exactly to the value added to the score in one of the steps
        # of the original removal process. Summing these column maximums gives the final score.
        for j in range(num_cols):
            # Initialize the maximum value found in the current column `j`.
            # Since constraints state 0 <= nums[i][j], initializing with 0 is safe.
            max_in_col = 0 
            
            # Iterate through each row `i` to find the element at column `j`.
            for i in range(num_rows):
                # Update `max_in_col` if the current element `nums[i][j]` is larger.
                max_in_col = max(max_in_col, nums[i][j])
            
            # Add the maximum value found in this column to the total score.
            total_score += max_in_col

        # Step 3: Return the final calculated score.
        return total_score