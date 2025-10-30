class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 10**9 + 7
        # K is the maximum number of non-empty stages (cannot exceed n or x)
        K = min(n, x)
        
        # Compute Stirling numbers of the second kind S(n, k) for k = 0..K
        # We'll do rolling DP since we only need the nth row.
        dp_prev = [0] * (K + 1)
        dp_prev[0] = 1  # S(0,0) = 1
        for i in range(1, n + 1):
            dp = [0] * (K + 1)
            # j cannot exceed i or K
            up = min(i, K)
            for j in range(1, up + 1):
                # S(i,j) = S(i-1, j-1) + j * S(i-1, j)
                dp[j] = (dp_prev[j-1] + j * dp_prev[j]) % mod
            dp_prev = dp
        
        # Precompute factorials and inverse factorials up to x for binomial coefficients
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = fact[i-1] * i % mod
        invfact = [1] * (x + 1)
        invfact[x] = pow(fact[x], mod-2, mod)
        for i in range(x, 0, -1):
            invfact[i-1] = invfact[i] * i % mod
        
        # Precompute powers of y up to K
        ypow = [1] * (K + 1)
        for i in range(1, K + 1):
            ypow[i] = ypow[i-1] * y % mod
        
        # Sum over k = 1..K:
        # ways to choose k stages: C(x, k)
        # ways to assign n performers onto exactly k non-empty stages: k! * S(n, k)
        # ways to assign scores: y^k
        ans = 0
        for k in range(1, K + 1):
            # C(x, k)
            cxk = fact[x] * invfact[k] % mod * invfact[x-k] % mod
            # number of surjections = k! * S(n, k)
            surj = fact[k] * dp_prev[k] % mod
            term = cxk * surj % mod * ypow[k] % mod
            ans = (ans + term) % mod
        
        return ans