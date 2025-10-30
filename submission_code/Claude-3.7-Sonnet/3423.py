class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        answer_sum = 0
        
        n = len(nums)
        dp = [0] * n
        
        # Initial DP calculation
        dp[0] = max(0, nums[0])
        
        for i in range(1, n):
            dp[i] = max(dp[i-1], (dp[i-2] if i > 1 else 0) + nums[i])
        
        # Process each query
        for pos, val in queries:
            nums[pos] = val
            
            # Recompute DP from the modified position onwards
            if pos == 0:
                dp[0] = max(0, nums[0])
            
            for i in range(max(1, pos), n):
                dp[i] = max(dp[i-1], (dp[i-2] if i > 1 else 0) + nums[i])
            
            answer_sum = (answer_sum + dp[n-1]) % MOD
        
        return answer_sum