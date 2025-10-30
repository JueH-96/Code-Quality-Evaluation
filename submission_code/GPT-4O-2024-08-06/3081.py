from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the middle of the array
        mid = n // 2
        # Initialize the minimum length as the total length
        min_length = n
        
        # Iterate over the first half of the array
        for i in range(mid):
            # Check if there is a valid pair (i, i + mid)
            if nums[i] < nums[i + mid]:
                # Calculate the remaining length after removing pairs
                min_length = min(min_length, n - 2 * (i + 1))
        
        return min_length