class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if (nums[i] + nums[j]) % 2 == 0:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)