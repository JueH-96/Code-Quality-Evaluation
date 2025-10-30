from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        # Iterate through possible 1-based indices from 1 to n
        for i in range(1, n + 1):
            # Check if the 1-based index i divides n
            if n % i == 0:
                # If i divides n, the element at 1-based index i is special.
                # In 0-based indexing, this element is at index i - 1.
                # Add the square of this element to the total sum.
                total_sum += nums[i - 1] * nums[i - 1]
        return total_sum