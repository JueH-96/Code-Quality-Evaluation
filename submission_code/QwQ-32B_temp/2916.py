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
                dp[i][j] = False
                for k in range(i, j):
                    left_sum = prefix[k + 1] - prefix[i]
                    right_sum = prefix[j + 1] - prefix[k + 1]
                    left_len = k - i + 1
                    right_len = j - k
                    left_ok = (left_len == 1) or (left_sum >= m)
                    right_ok = (right_len == 1) or (right_sum >= m)
                    if left_ok and right_ok:
                        if dp[i][k] and dp[k + 1][j]:
                            dp[i][j] = True
                            break  # No need to check further k's
        return dp[0][n - 1]