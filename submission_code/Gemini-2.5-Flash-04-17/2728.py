from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        """
        Calculates the final score by repeatedly removing the largest element
        from each row, and adding the maximum of the removed elements to the score.

        Args:
            nums: A 0-indexed 2D integer array.

        Returns:
            The final score.
        """
        # The problem can be solved by sorting each row.
        # After sorting each row in ascending order, the largest element
        # in a row is at the last index, the second largest at the second
        # to last index, and so on.
        # In each operation (round), we are taking the largest *remaining* element
        # from each row. If we sort ascending, the largest remaining elements
        # in the first round are at index len(row)-1 in each sorted row.
        # In the second round, they are at index len(row)-2, and so on.
        # The total number of rounds is equal to the number of columns.

        # Sort each row in ascending order
        for row in nums:
            row.sort()

        total_score = 0
        
        # Get the number of columns (assuming all rows have the same length
        # based on the problem description and examples).
        # The number of rounds is equal to the number of columns.
        cols = len(nums[0])
        rows = len(nums)

        # Iterate through the columns from right to left.
        # This corresponds to picking the largest element from each row
        # (index cols-1 after sorting), then the second largest (index cols-2), etc.
        for j in range(cols - 1, -1, -1):
            # In each round (represented by column index j), find the maximum
            # element among all elements at index j across all rows.
            max_val_in_round = 0

            for i in range(rows):
                max_val_in_round = max(max_val_in_round, nums[i][j])

            # Add the maximum value found in this round to the total score.
            total_score += max_val_in_round

        return total_score