class Solution:
    def waysToReachStair(self, k: int) -> int:
        memo = {}

        def solve(current_stair, current_jump, can_go_down):
            if (current_stair, current_jump, can_go_down) in memo:
                return memo[(current_stair, current_jump, can_go_down)]

            if current_stair == k:
                return 1
            if current_stair < 0:
                return 0

            ways = 0

            # Down operation
            if current_stair > 0 and can_go_down:
                ways += solve(current_stair - 1, current_jump, False)

            # Up operation
            ways += solve(current_stair + (1 << current_jump), current_jump + 1, True)

            memo[(current_stair, current_jump, can_go_down)] = ways
            return ways

        return solve(1, 0, True)