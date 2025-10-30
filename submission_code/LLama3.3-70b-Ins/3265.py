from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if abs(subarray[0] - subarray[-1]) == k:
                    max_sum = max(max_sum, sum(subarray))
        
        return max_sum