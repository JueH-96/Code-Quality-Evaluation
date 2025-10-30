from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)

        # Helper function to calculate the maximum sum with a given peak
        def calculate_sum(peak: int) -> int:
            left_max = 0
            right_max = 0
            current_height = 0

            # Calculate the maximum sum for the left part
            for i in range(peak, -1, -1):
                current_height = max(current_height, maxHeights[i])
                left_max += current_height

            current_height = 0

            # Calculate the maximum sum for the right part
            for i in range(peak, n):
                current_height = max(current_height, maxHeights[i])
                right_max += current_height

            # The total sum is the sum of the left and right parts minus the peak counted twice
            return left_max + right_max - current_height

        # Try each index as the peak and find the maximum sum
        max_sum = 0
        for i in range(n):
            max_sum = max(max_sum, calculate_sum(i))

        return max_sum