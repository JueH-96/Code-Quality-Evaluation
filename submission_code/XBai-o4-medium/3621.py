from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        for num in nums:
            if num < k:
                return -1
        
        # All elements are >= k. Now count unique elements greater than k.
        unique_nums = set(nums)
        count = 0
        for num in unique_nums:
            if num > k:
                count += 1
        return count