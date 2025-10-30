class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        dp = {}

        def solve(curr, jump):
            if (curr, jump) in dp:
                return dp[(curr, jump)]

            if curr == k:
                return 1
            if curr < 0 or curr > 2 * k:  # Optimization: Prune search space
                return 0

            ans = 0
            if curr > 0:
                ans = (ans + solve(curr - 1, jump)) % MOD
            ans = (ans + solve(curr + (1 << jump), jump + 1)) % MOD
            dp[(curr, jump)] = ans
            return ans

        return solve(1, 0)