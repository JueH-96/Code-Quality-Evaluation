class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_num = max(nums)
        
        dp = [[0] * (max_num + 1) for _ in range(n)]
        
        for i in range(nums[0] + 1):
            dp[0][i] = 1
        
        for i in range(1, n):
            prefix_sum = [0] * (max_num + 2)
            for j in range(max_num + 1):
                prefix_sum[j + 1] = (prefix_sum[j] + dp[i - 1][j]) % MOD
            
            for j in range(nums[i] + 1):
                dp[i][j] = prefix_sum[j + 1]
        
        result = sum(dp[n - 1]) % MOD
        return result