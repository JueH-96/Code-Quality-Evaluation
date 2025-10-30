class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        memo = {}

        def dp(stair, jump, prev_down):
            if (stair, jump, prev_down) in memo:
                return memo[(stair, jump, prev_down)]

            if stair == k:
                return 1

            if stair > k:
                return 0

            ways = 0
            # go up
            ways += dp(stair + 2**jump, jump + 1, False)

            # go down
            if not prev_down and stair > 0:
                ways += dp(stair - 1, jump, True)

            memo[(stair, jump, prev_down)] = ways % MOD
            return memo[(stair, jump, prev_down)]

        return dp(1, 0, False)