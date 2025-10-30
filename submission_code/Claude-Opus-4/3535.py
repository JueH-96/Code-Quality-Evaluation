class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] = number of ways where arr1[i] = j
        # Since nums[i] <= 50, arr1[i] can be at most 50
        dp = [[0] * 51 for _ in range(n)]
        
        # Base case: for position 0, arr1[0] can be any value from 0 to nums[0]
        for j in range(nums[0] + 1):
            dp[0][j] = 1
        
        # Fill the dp table
        for i in range(1, n):
            for j in range(nums[i] + 1):  # arr1[i] = j
                # arr1[i-1] must satisfy:
                # 1. arr1[i-1] <= j (non-decreasing)
                # 2. arr1[i-1] >= j - (nums[i] - nums[i-1]) (to ensure arr2 is non-increasing)
                min_prev = max(0, j - (nums[i] - nums[i-1]))
                max_prev = j
                
                for k in range(min_prev, max_prev + 1):
                    if k <= nums[i-1]:  # arr1[i-1] must be <= nums[i-1]
                        dp[i][j] = (dp[i][j] + dp[i-1][k]) % MOD
        
        # Sum all valid endings
        result = 0
        for j in range(51):
            result = (result + dp[n-1][j]) % MOD
        
        return result