from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # If k is large enough to reduce the array to a single element
        if k >= n - 1:
            return nums[0]  # The only element left after full reduction
        
        # Try to find the minimum OR by reducing the array
        min_or = float('inf')
        
        # We will try to reduce the array by making at most k operations
        for i in range(n - k):
            # Calculate the OR of the subarray from i to i+k
            current_or = nums[i]
            for j in range(i + 1, i + k + 1):
                current_or |= nums[j]
            # Update the minimum OR found
            min_or = min(min_or, current_or)
        
        return min_or