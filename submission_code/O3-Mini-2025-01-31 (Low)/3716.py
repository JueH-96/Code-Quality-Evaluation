from typing import List
from collections import defaultdict

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[j] is a dictionary mapping last difference 'd' to the maximum chain length
        dp = [defaultdict(lambda: 0) for _ in range(n)]
        ans = 1  # minimum subsequence length is 1 (we always have at least one element)
        
        for j in range(n):
            # for each prior index i 
            for i in range(j):
                d = abs(nums[j] - nums[i])
                # start a new chain from i to j: it has length 2
                # or extend any chain ending at i with last difference >= d.
                candidate = 2  # at least, from the pair (i,j)
                # try to extend any chain ending at i that permits this new difference
                for prev_diff, length in dp[i].items():
                    if prev_diff >= d:
                        candidate = max(candidate, length + 1)
                # update chain ending at j with difference d
                if candidate > dp[j][d]:
                    dp[j][d] = candidate
                    ans = max(ans, candidate)
                    
            # also consider single element chain (length=1) is already counted in ans.
        
        return ans