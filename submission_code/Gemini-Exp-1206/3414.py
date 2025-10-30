class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        dp = {}

        def solve(curr, jump, prev_down):
            if curr == k:
                return 1
            if (curr, jump, prev_down) in dp:
                return dp[(curr, jump, prev_down)]

            ans = 0
            if curr > 0 and not prev_down:
                ans += solve(curr - 1, jump, True)
            
            ans += solve(curr + (1 << jump), jump + 1, False)

            dp[(curr, jump, prev_down)] = ans
            return ans
        
        return solve(1, 0, False) * 2