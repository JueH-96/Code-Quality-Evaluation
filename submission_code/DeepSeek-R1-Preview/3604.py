class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        max_n = 1000
        
        if not hasattr(Solution, 'stirling'):
            Solution.stirling = [[0] * (max_n + 1) for _ in range(max_n + 1)]
            Solution.stirling[0][0] = 1
            for i in range(1, max_n + 1):
                for j in range(1, i + 1):
                    Solution.stirling[i][j] = (j * Solution.stirling[i-1][j] + Solution.stirling[i-1][j-1]) % MOD
            
            Solution.comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
            for x_ in range(max_n + 1):
                Solution.comb[x_][0] = 1
                for k_ in range(1, x_):
                    Solution.comb[x_][k_] = (Solution.comb[x_-1][k_-1] + Solution.comb[x_-1][k_]) % MOD
                Solution.comb[x_][x_] = 1
            
            Solution.fact = [1] * (max_n + 1)
            for i in range(1, max_n + 1):
                Solution.fact[i] = (Solution.fact[i-1] * i) % MOD
        
        sum_total = 0
        max_k = min(x, n)
        for k in range(1, max_k + 1):
            c = Solution.comb[x][k]
            s = Solution.stirling[n][k]
            f = Solution.fact[k]
            y_pow = pow(y, k, MOD)
            term = (c * s) % MOD
            term = (term * f) % MOD
            term = (term * y_pow) % MOD
            sum_total = (sum_total + term) % MOD
        
        return sum_total