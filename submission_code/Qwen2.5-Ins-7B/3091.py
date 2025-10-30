from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        cnt = Counter(nums)
        dp = [0] * (r + 1)
        dp[0] = 1
        
        for num, freq in cnt.items():
            for j in range(r, num - 1, -1):
                for k in range(1, freq + 1):
                    dp[j] = (dp[j] + dp[j - k * num]) % mod
        
        return sum(dp[l:r+1]) % mod