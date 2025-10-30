class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        memo = {}
        MOD = 10**9 + 7

        def dp(target, index):
            if (target, index) in memo:
                return memo[(target, index)]

            if target == 0:
                return 1
            if target < 0:
                return 0
            if index == 0:
                return 0

            res = dp(target, index - 1)
            power = index ** x
            if power <= target:
                res = (res + dp(target - power, index - 1)) % MOD

            memo[(target, index)] = res
            return res

        max_base = 0
        while (max_base + 1) ** x <= n:
            max_base += 1

        return dp(n, max_base)