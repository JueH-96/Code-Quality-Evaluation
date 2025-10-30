class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(n):
            for j in range(i + 1):
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD

        def is_k_reducible(num, k):
            if num == 1:
                return k >= 0
            if k == 0:
                return False
            
            count = 0
            while num > 0:
                if num % 2 == 1:
                    count += 1
                num //= 2
            return is_k_reducible(count, k - 1)

        ans = 0
        ones = 0
        for i in range(n):
            if s[i] == '0':
                continue
            for j in range(n + 1):
                if is_k_reducible(j + ones, k):
                    if i == n - 1 and j == 0:
                        continue
                    ans = (ans + dp[n - i - 1][j]) % MOD
            ones += 1

        if is_k_reducible(ones, k):
            ans = (ans + 1) % MOD
        
        if k == 1:
            ans = (ans - 1 + MOD) % MOD

        return ans