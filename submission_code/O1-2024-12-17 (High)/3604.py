class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to 1000
        max_val = 1000
        fac = [1]*(max_val+1)
        for i in range(1, max_val+1):
            fac[i] = (fac[i-1]*i) % MOD
        
        ifac = [1]*(max_val+1)
        ifac[max_val] = pow(fac[max_val], MOD-2, MOD)  # Fermat's little theorem
        for i in reversed(range(max_val)):
            ifac[i] = (ifac[i+1]*(i+1)) % MOD
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return (fac[n] * ((ifac[r]*ifac[n-r]) % MOD)) % MOD
        
        # Precompute powers of y up to 1000
        pow_y = [1]*(max_val+1)
        for i in range(1, max_val+1):
            pow_y[i] = (pow_y[i-1]*y) % MOD
        
        # Precompute Stirling numbers of the second kind S(n,k) for n,k up to 1000
        # S(n, k) = k*S(n-1, k) + S(n-1, k-1), with S(0,0)=1, S(n,0)=0 for n>0
        S = [[0]*(max_val+1) for _ in range(max_val+1)]
        S[0][0] = 1
        for i in range(1, max_val+1):
            for j in range(1, i+1):
                S[i][j] = (S[i-1][j-1] + j*S[i-1][j]) % MOD
        
        # Calculate the sum over k=1..min(n,x) of:
        #  C(x, k) * k! * S(n,k) * (y^k)
        limit = min(n, x)
        ans = 0
        for k in range(1, limit+1):
            term = comb(x, k)            # Choose which k stages are used
            term = (term * fac[k]) % MOD # Permute those k stages (they are labeled)
            term = (term * S[n][k]) % MOD# Distribute n performers onto those k stages surjectively
            term = (term * pow_y[k]) % MOD
            ans = (ans + term) % MOD
        
        return ans