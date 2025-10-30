from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)

        # Initialize the current height to be assigned
        current_height = len(maximumHeight)

        # Variable to store the total sum of heights
        total_sum = 0

        # Assign heights and calculate the sum
        for max_height in maximumHeight:
            if current_height > max_height:
                return -1
            total_sum += current_height
            current_height -= 1

        return total_sum