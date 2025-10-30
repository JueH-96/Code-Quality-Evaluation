class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        """
        Calculates the score based on a matrix operation game.

        The core idea is to realize that the process of repeatedly taking the largest
        element from each row can be simplified by sorting each row first.

        1. Sort each row: By sorting each row in ascending order, the largest element
           will be at the end, the second largest will be second to last, and so on.
           For example, `[7, 2, 1]` becomes `[1, 2, 7]`.

        2. Simulate the operations column-wise: After sorting, the first operation
           (picking the largest from each row) corresponds to picking every element
           in the last column of the matrix. The problem states we should add the
           maximum of these picked numbers to the score. This is simply the maximum
           value in the last column.

        3. The second operation corresponds to picking every element in the
           second-to-last column, and the score increases by the maximum of that column.
           This continues for all columns.

        4. The final score is the sum of the maximums of each column of the
           row-sorted matrix.

        This can be implemented efficiently by first sorting each row, then transposing
        the matrix (or iterating column by column) and summing the maximums. The
        `zip(*nums)` idiom in Python is a concise way to transpose a 2D list.
        """
        
        # Step 1: Sort each row of the matrix. This has a time complexity of
        # O(m * n * log n), where m is rows and n is columns.
        for row in nums:
            row.sort()
            
        # Step 2: Use zip(*nums) to transpose the matrix, effectively iterating through columns.
        # For each column, find the maximum value.
        # Sum these maximums to get the final score.
        # This part has a time complexity of O(m * n).
        # The overall complexity is dominated by the sorting step.
        return sum(max(col) for col in zip(*nums))