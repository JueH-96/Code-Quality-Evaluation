class Solution:
    def waysToReachStair(self, k: int) -> int:
        memo = {}

        def solve(curr, jump, last_down):
            if curr == k:
                return 1
            if curr < 0 or curr > k + 30:
                return 0

            if (curr, jump, last_down) in memo:
                return memo[(curr, jump, last_down)]

            ways = 0
            
            # Option 1: Go down
            if curr > 0 and not last_down:
                ways += solve(curr - 1, jump, True)

            # Option 2: Go up
            ways += solve(curr + (1 << jump), jump + 1, False)

            memo[(curr, jump, last_down)] = ways
            return ways

        return solve(1, 0, False)