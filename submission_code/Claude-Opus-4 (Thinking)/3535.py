class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] = number of ways to fill positions 0 to i such that arr1[i] = j
        dp = [[0] * 51 for _ in range(n)]
        
        # Base case: fill position 0
        for j in range(nums[0] + 1):
            dp[0][j] = 1
        
        # Fill remaining positions
        for i in range(1, n):
            # Compute prefix sums for dp[i-1]
            prefix = [0] * 52
            for j in range(51):
                prefix[j+1] = (prefix[j] + dp[i-1][j]) % MOD
            
            diff = max(0, nums[i] - nums[i-1])
            for j in range(nums[i] + 1):
                # arr1[i] = j, we need arr1[i-1] <= j - diff
                max_k = min(nums[i-1], j - diff)
                if max_k >= 0:
                    dp[i][j] = prefix[max_k + 1]
        
        # Sum all valid configurations for the last position
        result = 0
        for j in range(nums[n-1] + 1):
            result = (result + dp[n-1][j]) % MOD
        
        return result