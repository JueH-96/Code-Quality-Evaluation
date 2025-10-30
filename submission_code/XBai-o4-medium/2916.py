from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                for k in range(i, j):
                    # Check if splitting at k is valid
                    left_ok = True
                    if k != i:
                        sum_left = prefix[k + 1] - prefix[i]
                        if sum_left < m:
                            left_ok = False
                    right_ok = True
                    if k + 1 != j:
                        sum_right = prefix[j + 1] - prefix[k + 1]
                        if sum_right < m:
                            right_ok = False
                    if left_ok and right_ok:
                        if dp[i][k] and dp[k + 1][j]:
                            dp[i][j] = True
                            break
        
        return dp[0][n - 1]