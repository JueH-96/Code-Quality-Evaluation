class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_num = max(nums)
        
        # dp[i][j] represents the number of good subsequences ending at value j
        # at position i in the array
        dp = [[0] * (max_num + 1) for _ in range(n)]
        
        # Initialize for first position
        dp[0][nums[0]] = 1
        
        # For each position in array
        for i in range(1, n):
            curr = nums[i]
            # Copy previous state
            for j in range(max_num + 1):
                dp[i][j] = dp[i-1][j]
            
            # Add single element subsequence
            dp[i][curr] = (dp[i][curr] + 1) % MOD
            
            # For each previous value
            for prev in range(max_num + 1):
                if dp[i-1][prev] > 0:
                    # If absolute difference is 1
                    if abs(prev - curr) == 1:
                        dp[i][curr] = (dp[i][curr] + dp[i-1][prev]) % MOD
        
        total = 0
        # Sum up all subsequences from the last position
        for val in range(max_num + 1):
            total = (total + dp[n-1][val]) % MOD
        
        return total