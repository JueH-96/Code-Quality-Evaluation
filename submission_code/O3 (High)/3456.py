from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][c] = longest length of a good subsequence that
        #            ends at position i (takes nums[i]) and
        #            uses exactly c changes so far.
        dp = [[0]*(k+1) for _ in range(n)]
        ans = 1                              # at least one element can always be taken
        
        for i in range(n):
            dp[i][0] = 1                     # subsequence consisting of nums[i] only
            for j in range(i):               # try to extend subsequences ending at j
                for c in range(k+1):
                    if dp[j][c] == 0:        # nothing to extend
                        continue
                    new_c = c + (nums[j] != nums[i])
                    if new_c <= k:
                        dp[i][new_c] = max(dp[i][new_c], dp[j][c] + 1)
            ans = max(ans, max(dp[i]))
        
        return ans