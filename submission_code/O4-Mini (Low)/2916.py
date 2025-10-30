from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # prefix sums for O(1) range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # helper to get sum of nums[l..r] inclusive
        def range_sum(l: int, r: int) -> int:
            return prefix[r+1] - prefix[l]
        
        # dp[i][j] = True if subarray nums[i..j] can be fully split into singletons
        dp = [[False] * n for _ in range(n)]
        
        # length-1 segments are trivially "fully split" (they're already singletons)
        for i in range(n):
            dp[i][i] = True
        
        # consider all lengths from 2 up to n
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                # try every possible split position k between i..j-1
                for k in range(i, j):
                    # left = nums[i..k], right = nums[k+1..j]
                    left_len = k - i + 1
                    right_len = j - (k+1) + 1
                    left_sum = range_sum(i, k)
                    right_sum = range_sum(k+1, j)
                    
                    # check the split condition for both sides
                    left_ok = (left_len == 1) or (left_sum >= m)
                    right_ok = (right_len == 1) or (right_sum >= m)
                    
                    if left_ok and right_ok and dp[i][k] and dp[k+1][j]:
                        dp[i][j] = True
                        break
        
        return dp[0][n-1]