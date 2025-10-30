class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] = number of subsequences from nums[0:i] that sum to j
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # empty subsequence sums to 0
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                # Don't include nums[i-1]
                dp[i][j] = dp[i-1][j]
                
                # Include nums[i-1] if possible
                if j >= nums[i-1]:
                    dp[i][j] = (dp[i][j] + dp[i-1][j - nums[i-1]]) % MOD
        
        # Now calculate the contribution of each subsequence
        # We need to track which elements are used in each subsequence
        # Let's use a different approach
        
        result = 0
        
        # For each subset mask
        for mask in range(1, 1 << n):
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
            
            # Count subsequences of this subset that sum to k
            m = len(subset)
            subset_dp = [[0] * (k + 1) for _ in range(m + 1)]
            subset_dp[0][0] = 1
            
            for i in range(1, m + 1):
                for j in range(k + 1):
                    subset_dp[i][j] = subset_dp[i-1][j]
                    if j >= subset[i-1]:
                        subset_dp[i][j] = (subset_dp[i][j] + subset_dp[i-1][j - subset[i-1]]) % MOD
            
            # Add contribution of this subset
            count = subset_dp[m][k]
            contribution = (count * pow(2, n - m, MOD)) % MOD
            result = (result + contribution) % MOD
        
        return result