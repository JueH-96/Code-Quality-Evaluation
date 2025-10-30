from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        
        for i in range(n):
            count = 1
            for j in range(i - 1, -1, -1):
                if nums[j] == nums[i]:
                    count += 1
                else:
                    break
            dp[i] = count + (dp[j] if j >= 0 else 0)
            for j in range(i - 1, -1, -1):
                if nums[j] == nums[i]:
                    dp[i] = max(dp[i], dp[j] + (dp[i - 1] if i > 0 else 0) - 1)
                else:
                    break
            dp[i] = min(dp[i], k + 1)
        
        return max(dp)