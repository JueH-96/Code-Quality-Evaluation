from collections import Counter
from typing import List

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        count = Counter(nums)
        unique_nums = sorted(count.keys())
        n = len(unique_nums)
        
        if 0 in count:
            zero_count = count[0]
            del count[0]
            unique_nums.remove(0)
            n -= 1
        
        dp = [0] * (r + 1)
        dp[0] = 1
        
        for num in unique_nums:
            next_dp = dp[:]
            for i in range(num, r + 1):
                total = 0
                for k in range(1, count[num] + 1):
                    if i - k * num < 0:
                        break
                    total = (total + dp[i - k * num]) % MOD
                next_dp[i] = (next_dp[i] + total) % MOD
            dp = next_dp
        
        result = sum(dp[l:r + 1]) % MOD
        
        if zero_count > 0:
            result = (result * (zero_count + 1)) % MOD
        
        return result