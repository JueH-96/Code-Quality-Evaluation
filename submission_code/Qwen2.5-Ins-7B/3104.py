from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ways = 0
        
        if 0 not in nums:
            ways += 1
        
        for i in range(n):
            selected = i + 1
            if selected > nums[i] and (i == n - 1 or selected < nums[i + 1]):
                ways += 1
        
        return ways