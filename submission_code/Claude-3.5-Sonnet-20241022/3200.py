class Solution:
    def stringCount(self, n: int) -> int:
        if n < 4:
            return 0
            
        MOD = 10**9 + 7
        
        # dp[i][l][e][t] represents number of ways to form string of length i
        # with l L's, e E's and t T's remaining to form "leet"
        dp = {}
        
        def solve(pos, l, e, t):
            if pos == n:
                return 1 if l == 0 and e == 0 and t == 0 else 0
                
            state = (pos, l, e, t)
            if state in dp:
                return dp[state]
                
            ans = 0
            # Add 'l'
            if l > 0:
                ans = (ans + solve(pos + 1, l - 1, e, t)) % MOD
            # Add 'e'    
            if e > 0:
                ans = (ans + solve(pos + 1, l, e - 1, t)) % MOD
            # Add 't'    
            if t > 0:
                ans = (ans + solve(pos + 1, l, e, t - 1)) % MOD
            # Add any other lowercase letter (22 choices excluding l,e,t)    
            ans = (ans + (22 * solve(pos + 1, l, e, t)) % MOD) % MOD
            
            dp[state] = ans
            return ans
            
        return solve(0, 1, 2, 1)