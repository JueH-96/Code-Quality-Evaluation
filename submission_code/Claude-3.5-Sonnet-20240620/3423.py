class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        def update_dp():
            nonlocal dp
            for i in range(2, n):
                dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        dp = [0] * n
        dp[0] = max(0, nums[0])
        if n > 1:
            dp[1] = max(dp[0], nums[1])
        update_dp()
        
        total_sum = 0
        for pos, x in queries:
            nums[pos] = x
            if pos == 0:
                dp[0] = max(0, nums[0])
                if n > 1:
                    dp[1] = max(dp[0], nums[1])
            elif pos == 1:
                dp[1] = max(dp[0], nums[1])
            update_dp()
            total_sum = (total_sum + dp[-1]) % MOD
        
        return total_sum