from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_num = max(nums)
        dp = [[0] * (max_num + 1) for _ in range(n)]
        
        for i in range(n):
            for j in range(max_num + 1):
                if i == 0:
                    dp[i][j] = 1
                else:
                    for k in range(j + 1):
                        dp[i][j] += dp[i - 1][k]
                        dp[i][j] %= MOD
        
        result = 0
        for i in range(max_num + 1):
            result += dp[n - 1][i] * dp[n - 1][nums[0] - i]
            result %= MOD
        
        for i in range(1, n):
            new_result = 0
            for j in range(max_num + 1):
                for k in range(j + 1):
                    new_result += dp[i - 1][k] * dp[n - i - 1][nums[i] - j]
                    new_result %= MOD
            result = new_result
        
        return result