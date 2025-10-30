class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute Stirling numbers of the second kind: S[i][j]
        # S[i][j] = number of ways to partition i items into j non-empty unlabeled subsets
        S = [[0] * (n+1) for _ in range(n+1)]
        S[0][0] = 1
        for i in range(1, n+1):
            for j in range(1, i+1):
                # recurrence: S(i,j) = S(i-1, j-1) + j * S(i-1, j)
                S[i][j] = (S[i-1][j-1] + j * S[i-1][j]) % MOD
        
        # Precompute factorials and inverse factorials up to max(n, x)
        m = max(n, x)
        fac = [1] * (m+1)
        ifac = [1] * (m+1)
        for i in range(1, m+1):
            fac[i] = fac[i-1] * i % MOD
        # Fermat inverse factorial
        ifac[m] = pow(fac[m], MOD-2, MOD)
        for i in range(m, 0, -1):
            ifac[i-1] = ifac[i] * i % MOD
        
        # Helper: P(x,k) = x! / (x-k)!  (number of ways to choose and order k distinct stages)
        def perm(x, k):
            if k > x:
                return 0
            return fac[x] * ifac[x-k] % MOD
        
        ans = 0
        # Sum over k = number of occupied stages
        # For each k from 1..min(n,x):
        #   - choose k distinct stages in order: P(x,k)
        #   - assign n performers onto those k stages surjectively: k! * S[n][k]
        #   - award scores to each band: y^k
        # But P(x,k) * (k! * S[n][k]) = P(x,k) * k! * S[n][k]
        # We can just compute perm(x,k) * (k! * S[n][k]) * y^k.
        for k in range(1, min(n, x) + 1):
            ways_assign = perm(x, k)  # x!/(x-k)!
            ways_group = fac[k] * S[n][k] % MOD
            ways_score = pow(y, k, MOD)
            ans = (ans + ways_assign * ways_group % MOD * ways_score) % MOD
        
        return ans