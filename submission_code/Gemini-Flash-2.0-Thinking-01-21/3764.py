from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        """
        Finds the maximum sum of at most k elements from the grid,
        respecting the limits on the number of elements per row.

        Args:
            grid: A 2D list of integers.
            limits: A list of integers representing the maximum number of elements
                    allowed from each row.
            k: The maximum total number of elements to pick from the grid.

        Returns:
            The maximum possible sum.
        """
        # List to hold all eligible elements from the grid
        # An element is eligible if it is among the top limits[i] elements
        # in row i (after sorting row i descending).
        eligible_elements = []

        # Process each row
        for i in range(len(grid)):
            # Sort the current row in descending order to easily access the largest elements
            grid[i].sort(reverse=True)

            # Collect the first limits[i] elements from the sorted row.
            # These are the largest elements allowed from this row.
            # The loop range is safe because 0 <= limits[i] <= m = len(grid[i]).
            for j in range(limits[i]):
                eligible_elements.append(grid[i][j])

        # We want to pick at most k elements from the combined list
        # of eligible elements to maximize the sum.
        # This is achieved by picking the largest elements from this list.
        # Sort the combined list of eligible elements in descending order.
        eligible_elements.sort(reverse=True)

        # Calculate the sum of the top k elements from the sorted eligible list.
        # Based on the constraint 0 <= k <= min(n * m, sum(limits)), and
        # the size of eligible_elements is exactly sum(limits), we have
        # k <= len(eligible_elements).
        # So we can just take the first k elements.
        # Using slicing is concise and handles k=0 correctly.
        max_sum = sum(eligible_elements[:k])

        return max_sum