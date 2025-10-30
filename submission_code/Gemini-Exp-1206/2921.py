class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7
        
        def count(s):
            n = len(s)
            dp = [[0] * 10 for _ in range(n + 1)]
            
            for i in range(1, 10):
                dp[1][i] = 1
            
            for i in range(2, n + 1):
                for j in range(10):
                    if j > 0:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % mod
                    if j < 9:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % mod
            
            ans = 0
            for i in range(1, n):
                for j in range(10):
                    ans = (ans + dp[i][j]) % mod
            
            prefix = 0
            for i, c in enumerate(s):
                digit = int(c)
                for j in range((i == 0), digit):
                    if i == 0 or abs(j - int(s[i - 1])) == 1:
                        ans = (ans + dp[n - i][j]) % mod
                
                if i > 0 and abs(digit - int(s[i - 1])) != 1:
                    break
                
                if i == n - 1:
                    ans = (ans + 1) % mod
            
            return ans
        
        def is_stepping(s):
            for i in range(len(s) - 1):
                if abs(int(s[i]) - int(s[i + 1])) != 1:
                    return False
            return True

        ans = (count(high) - count(low) + mod) % mod
        if is_stepping(low):
            ans = (ans + 1) % mod
        
        return ans