from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        dp = [0] * (r + 1)
        dp[0] = 1  # empty set
        
        for num in freq:
            f = freq[num]
            k = 1
            while k <= f:
                for s in range(r, num - 1, -1):
                    if s - k * num >= 0:
                        dp[s] = (dp[s] + dp[s - k * num]) % MOD
                f -= k
                k *= 2
            if f > 0:
                for s in range(r, num - 1, -1):
                    if s - f * num >= 0:
                        dp[s] = (dp[s] + dp[s - f * num]) % MOD
        
        # Summing dp[s] for s from l to r
        result = 0
        for s in range(l, r + 1):
            if s <= r:
                result = (result + dp[s]) % MOD
        return result