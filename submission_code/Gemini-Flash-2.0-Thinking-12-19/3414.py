class Solution:
    def waysToReachStair(self, k: int) -> int:
        memo = {}
        max_jump = 40
        max_stair_offset = 200

        def solve(stair, jump, last_down):
            if stair == k:
                return 1
            if stair < 0 or stair > k + max_stair_offset or jump > max_jump:
                return 0
            if (stair, jump, last_down) in memo:
                return memo[(stair, jump, last_down)]
            
            ways = 0
            # Down operation
            if stair > 0 and not last_down:
                ways += solve(stair - 1, jump, True)
            # Up operation
            ways += solve(stair + (1 << jump), jump + 1, False)
            
            memo[(stair, jump, last_down)] = ways
            return ways

        return solve(1, 0, False)