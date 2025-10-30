from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Base case: empty subset
        
        for x, count in freq.items():
            # Iterate backwards to prevent overcounting
            for s in range(max_sum, -1, -1):
                if dp[s] == 0:
                    continue
                # Add 1 to count copies of x
                for k in range(1, count + 1):
                    new_s = s + k * x
                    if new_s > max_sum:
                        break
                    dp[new_s] = (dp[new_s] + dp[s]) % MOD
        
        # Sum all possible sums from l to r inclusive
        total = sum(dp[l:r+1]) % MOD
        return total