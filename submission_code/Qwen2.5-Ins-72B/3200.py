class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j][k][l] represents the number of good strings of length i
        # with at least j 'l's, k 'e's, and l 't's.
        dp = [[[[-1] * 2 for _ in range(3)] for _ in range(3)] for _ in range(n + 1)]
        
        def dfs(i, l, e, t):
            if i == n:
                return 1 if l == 1 and e == 2 and t == 1 else 0
            if dp[i][l][e][t] != -1:
                return dp[i][l][e][t]
            
            count = 0
            for char in range(26):
                if char == ord('l') - ord('a'):
                    count += dfs(i + 1, min(l + 1, 1), e, t)
                elif char == ord('e') - ord('a'):
                    count += dfs(i + 1, l, min(e + 1, 2), t)
                elif char == ord('t') - ord('a'):
                    count += dfs(i + 1, l, e, min(t + 1, 1))
                else:
                    count += dfs(i + 1, l, e, t)
            dp[i][l][e][t] = count % MOD
            return dp[i][l][e][t]
        
        return dfs(0, 0, 0, 0)