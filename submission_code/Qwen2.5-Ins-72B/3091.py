from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        count = Counter(nums)
        dp = [1] + [0] * r
        
        for num, freq in count.items():
            if num == 0:
                continue
            new_dp = [0] * (r + 1)
            for i in range(num, r + 1):
                new_dp[i] = dp[i - num]
            for i in range(r + 1):
                new_dp[i] = (new_dp[i] + dp[i]) % MOD
            for i in range(num * freq, r + 1):
                new_dp[i] = (new_dp[i] - dp[i - num * freq]) % MOD
            dp = new_dp
        
        result = sum(dp[l:r + 1]) % MOD
        if 0 in count:
            result = (result + count[0] * sum(dp[l - 1:r]) % MOD + (l == 0)) % MOD
        
        return result