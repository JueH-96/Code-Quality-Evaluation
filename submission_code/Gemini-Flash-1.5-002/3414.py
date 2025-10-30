class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        
        mod = 10**9 + 7
        dp = {}
        dp[0] = 2
        dp[1] = 4

        def solve(curr):
            if curr in dp:
                return dp[curr]
            
            ans = 0
            jump = 0
            while True:
                next_stair = curr + (2**jump)
                if next_stair > k + 10:
                    break
                ans = (ans + solve(next_stair -1)) % mod
                jump +=1
            
            
            if curr > 0:
                ans = (ans + solve(curr -1)) % mod
            
            dp[curr] = ans
            return ans

        return solve(k)