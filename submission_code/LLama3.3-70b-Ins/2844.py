from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Iterate over the indices of the array (1-indexed)
        for i in range(1, n + 1):
            # Check if the index divides the length of the array
            if n % i == 0:
                # If it does, add the square of the element at that index to the total sum
                total_sum += nums[i - 1] ** 2
        
        return total_sum