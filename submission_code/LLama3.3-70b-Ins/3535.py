from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        memo = {}

        def dp(i, prev1, prev2):
            if i == n:
                return 1
            if (i, prev1, prev2) in memo:
                return memo[(i, prev1, prev2)]
            res = 0
            for a in range(nums[i] + 1):
                b = nums[i] - a
                if a >= prev1 and b <= prev2:
                    res += dp(i + 1, a, b)
                    res %= MOD
            memo[(i, prev1, prev2)] = res
            return res

        return dp(0, -1, 51)