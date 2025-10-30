from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[j][c] = max length of a good subsequence ending at index j with cost c
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # A subsequence of length 1 (just nums[j]) has cost 0
        for j in range(n):
            dp[j][0] = 1
        
        ans = 1  # At least one element can always be picked
        
        for j in range(n):
            for i in range(j):
                for c in range(k + 1):
                    if dp[i][c] == 0:
                        continue
                    # cost of linking nums[i] -> nums[j]
                    if nums[i] == nums[j]:
                        # no extra cost
                        if dp[i][c] + 1 > dp[j][c]:
                            dp[j][c] = dp[i][c] + 1
                            ans = max(ans, dp[j][c])
                    else:
                        # costs 1 if we change value
                        if c + 1 <= k and dp[i][c] + 1 > dp[j][c + 1]:
                            dp[j][c + 1] = dp[i][c] + 1
                            ans = max(ans, dp[j][c + 1])
        
        return ans