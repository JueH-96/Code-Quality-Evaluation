class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        m_max = min(x, n)
        
        # Compute Stirling numbers up to the given n
        stir = [[0] * (n + 1) for _ in range(n + 1)]
        stir[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                stir[i][j] = (j * stir[i-1][j] + stir[i-1][j-1]) % MOD
        
        total = 0
        for m in range(1, m_max + 1):
            # Calculate permutation P(x, m)
            perm = 1
            for i in range(m):
                perm = perm * (x - i) % MOD
            # Get Stirling number S(n, m)
            s = stir[n][m]
            # Calculate y^m mod MOD
            y_pow = pow(y, m, MOD)
            # Accumulate the total
            total = (total + perm * s % MOD * y_pow) % MOD
        
        return total