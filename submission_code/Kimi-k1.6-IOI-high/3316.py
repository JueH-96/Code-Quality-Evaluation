from typing import List
from collections import defaultdict

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        # Initialize DP table: dp[j][c] is a dictionary of {min_diff: count}
        dp = [[defaultdict(int) for _ in range(k + 1)] for __ in range(n)]
        
        for j in range(n):
            for i in range(j):
                d = nums[j] - nums[i]
                # Update subsequences of length 2 ending at j
                dp[j][2][d] = (dp[j][2][d] + 1) % MOD
                # Update subsequences of longer lengths
                for c in range(1, k):
                    for md, cnt in dp[i][c].items():
                        new_md = min(md, d)
                        dp[j][c + 1][new_md] = (dp[j][c + 1][new_md] + cnt) % MOD
        
        total = 0
        for j in range(n):
            for md, cnt in dp[j][k].items():
                total = (total + md * cnt) % MOD
        return total