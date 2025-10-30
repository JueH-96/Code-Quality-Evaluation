class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] will store the number of ways to form arr1 up to index i with arr1[i] = j
        dp = [[0] * 51 for _ in range(n)]
        
        # Initialize the first element
        for j in range(nums[0] + 1):
            dp[0][j] = 1
        
        # Fill the dp table
        for i in range(1, n):
            prefix_sum = [0] * 51
            for j in range(51):
                prefix_sum[j] = dp[i-1][j] + (prefix_sum[j-1] if j > 0 else 0)
                prefix_sum[j] %= MOD
            
            for j in range(nums[i] + 1):
                dp[i][j] = prefix_sum[j]
        
        # Calculate the result
        result = 0
        for j in range(51):
            result = (result + dp[n-1][j]) % MOD
        
        return result