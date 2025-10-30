from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        
        for num in nums:
            for s in range(k, -1, -1):
                for m in range(n, -1, -1):
                    if dp[s][m]:
                        new_s = s + num
                        new_m = m + 1
                        if new_s <= k:
                            dp[new_s][new_m] = (dp[new_s][new_m] + dp[s][m]) % MOD
        
        result = 0
        for m in range(1, n + 1):
            result = (result + dp[k][m] * pow(2, n - m, MOD)) % MOD
        
        return result