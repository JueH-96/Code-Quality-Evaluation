class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate combinations using Pascal's triangle
        comb = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            comb[i][0] = comb[i][i] = 1
            for j in range(1, i):
                comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % MOD
        
        # Calculate powers
        pow_x = [1] * (n + 1)
        pow_y = [1] * (n + 1)
        for i in range(1, n + 1):
            pow_x[i] = (pow_x[i - 1] * x) % MOD
            pow_y[i] = (pow_y[i - 1] * y) % MOD
        
        ans = 0
        for k in range(1, min(n, x) + 1):
            term = (comb[n - 1][k - 1] * pow_x[k]) % MOD
            term = (term * pow_y[k]) % MOD
            
            # Stirling numbers of the second kind
            stirling = 0
            for j in range(k + 1):
                stirling_term = (comb[k][j] * pow(j, n, MOD)) % MOD
                if (k - j) % 2 == 0:
                    stirling = (stirling + stirling_term) % MOD
                else:
                    stirling = (stirling - stirling_term + MOD) % MOD
            
            factorial = 1
            for i in range(1, k + 1):
                factorial = (factorial * i) % MOD
            
            stirling = (stirling * pow(factorial, MOD - 2, MOD)) % MOD
            
            term = (term * stirling) % MOD
            ans = (ans + term) % MOD
        
        return ans