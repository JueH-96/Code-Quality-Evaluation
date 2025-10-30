from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if j == m:
                return 0 if i == n else float('inf')
            res = float('inf')
            for k in range(i, n):
                and_val = nums[i]
                for x in range(i, k + 1):
                    and_val &= nums[x]
                if and_val == andValues[j]:
                    res = min(res, nums[k] + dp(k + 1, j + 1))
            memo[(i, j)] = res
            return res

        res = dp(0, 0)
        return res if res != float('inf') else -1