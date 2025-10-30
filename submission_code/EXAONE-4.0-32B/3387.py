from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        m = n // 2
        cost = 0
        
        for i in range(m):
            if nums[i] > k:
                cost += nums[i] - k
                
        cost += abs(nums[m] - k)
        
        for i in range(m + 1, n):
            if nums[i] < k:
                cost += k - nums[i]
                
        return cost