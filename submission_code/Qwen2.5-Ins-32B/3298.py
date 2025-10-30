from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = [1] * len(nums)
        idx_map = {nums[0]: 0}
        
        for i in range(1, len(nums)):
            num = nums[i]
            if num in idx_map:
                dp[i] = dp[idx_map[num]] + 1
            if num - 1 in idx_map:
                dp[i] = max(dp[i], dp[idx_map[num - 1]] + 1)
            idx_map[num] = i
        
        return max(dp)