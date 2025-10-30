class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count(n):
            dp = [[0] * 10 for _ in range(len(n) + 1)]
            for i in range(10):
                dp[1][i] = 1
            for length in range(2, len(n) + 1):
                for digit in range(10):
                    if digit > 0:
                        dp[length][digit] += dp[length - 1][digit - 1]
                    if digit < 9:
                        dp[length][digit] += dp[length - 1][digit + 1]
                    dp[length][digit] %= MOD
            
            total = sum(dp[len(n)][i] for i in range(10)) % MOD
            for i in range(len(n) - 1, -1, -1):
                digit = int(n[i])
                if i == len(n) - 1:
                    total -= sum(dp[i + 1][j] for j in range(digit + 1, 10)) % MOD
                else:
                    total -= sum(dp[i + 1][j] for j in range(digit + 1, 10) if abs(j - int(n[i + 1])) == 1) % MOD
                if i == 0 and digit == 0:
                    total += 1
            return total % MOD
        
        return (count(high) - count(str(int(low) - 1))) % MOD