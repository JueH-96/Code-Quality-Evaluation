class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [-float('inf')] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            for j in range(i):
                if (nums[i] % 2 == nums[j] % 2):
                    dp[i] = max(dp[i], dp[j] + nums[i])
                else:
                    dp[i] = max(dp[i], dp[j] + nums[i] - x)
        
        return max(dp)