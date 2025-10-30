from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize max_diff to 0. Since absolute differences are non-negative,
        # 0 is a safe initial value. The problem guarantees n >= 2, so there
        # will always be at least one pair to compare.
        max_diff = 0
        
        # Iterate through the array to find the maximum absolute difference
        # between adjacent elements in the "linear" sense (i.e., nums[i] and nums[i+1]).
        # The loop runs up to n-2 because nums[i+1] would be nums[n-1] when i is n-2.
        for i in range(n - 1):
            diff = abs(nums[i] - nums[i+1])
            if diff > max_diff:
                max_diff = diff
                
        # For a circular array, the last element (nums[n-1]) is also adjacent
        # to the first element (nums[0]). Calculate this difference separately
        # and update max_diff if it's greater.
        circular_diff = abs(nums[n-1] - nums[0])
        if circular_diff > max_diff:
            max_diff = circular_diff
            
        return max_diff