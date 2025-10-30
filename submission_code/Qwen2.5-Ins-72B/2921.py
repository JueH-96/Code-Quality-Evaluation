class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count(n: str) -> int:
            n = '0' * (100 - len(n)) + n
            dp = [[[-1] * 10 for _ in range(2)] for _ in range(101)]
            
            def dfs(pos: int, tight: int, last: int) -> int:
                if pos == len(n):
                    return 1
                if dp[pos][tight][last] != -1:
                    return dp[pos][tight][last]
                
                limit = int(n[pos]) if tight else 9
                res = 0
                for digit in range(limit + 1):
                    if last == -1 or abs(digit - last) == 1:
                        new_tight = tight and digit == limit
                        res = (res + dfs(pos + 1, new_tight, digit)) % MOD
                
                dp[pos][tight][last] = res
                return res
            
            return dfs(0, 1, -1) - 1  # Subtract 1 to exclude the number 0
        
        return (count(high) - count(str(int(low) - 1))) % MOD