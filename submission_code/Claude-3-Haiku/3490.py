class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        dp = [1] * n
        max_len = 1
        
        for i in range(1, n):
            if (nums[i-1] + nums[i]) % 2 == 0:
                dp[i] = dp[i-1] + 1
                max_len = max(max_len, dp[i])
            else:
                dp[i] = 1
        
        return max_len