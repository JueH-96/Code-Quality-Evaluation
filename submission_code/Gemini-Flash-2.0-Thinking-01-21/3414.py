class Solution:
    def waysToReachStair(self, k: int) -> int:
        memo = {}
        def solve(current_stair, jump, last_down):
            if current_stair == k: return 1
            if current_stair < 0: return 0
            if current_stair > k + 20: return 0

            state = (current_stair, jump, last_down)
            if state in memo: return memo[state]

            ans = 0
            if current_stair > 0 and not last_down:
                ans += solve(current_stair - 1, jump, True)
            ans += solve(current_stair + (1 << jump), jump + 1, False)
            memo[state] = ans
            return ans
        return solve(1, 0, False)