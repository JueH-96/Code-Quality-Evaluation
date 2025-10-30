class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Initialize the dp arrays
        n = len(nums)
        dp = [0] * n
        dp[0] = max(0, nums[0])
        if n > 1:
            dp[1] = max(dp[0], nums[1])
        
        # Precompute the dp array
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        # Function to update the dp array after a change
        def update_dp(pos, new_val):
            nums[pos] = new_val
            # Recompute dp[pos] and its affected positions
            # Since changing nums[pos] affects dp[pos], dp[pos+1], etc.
            # We need to recompute from pos to the end
            for i in range(pos, n):
                if i == 0:
                    dp[i] = max(0, nums[i])
                elif i == 1:
                    dp[i] = max(dp[0], nums[i])
                else:
                    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        total = 0
        for pos, x in queries:
            update_dp(pos, x)
            total = (total + dp[-1]) % MOD
        
        return total