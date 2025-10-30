class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        N = len(nums)
        dp = [0] * (N + 1)
        
        # Compute initial DP array
        for i in range(1, N + 1):
            if i == 1:
                dp[i] = max(0, nums[i-1])
            else:
                dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
        
        total_sum = 0
        for query in queries:
            pos, x = query
            # Update nums at pos
            nums[pos] = x
            # Recompute DP from pos to N
            prev = dp[pos]
            prev_prev = dp[pos - 1] if pos >= 1 else 0
            for i in range(pos, N):
                temp = dp[i]
                dp[i] = max(prev, prev_prev + nums[i])
                prev_prev = prev
                prev = dp[i]
            dp[N] = prev
            # Add the new dp[N] to the total sum
            total_sum = (total_sum + dp[N]) % MOD
        
        return total_sum