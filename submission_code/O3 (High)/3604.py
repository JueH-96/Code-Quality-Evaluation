MOD = 10 ** 9 + 7

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        """
        Required value:
            Σ_{k=1..min(n,x)}  C(x,k) · k! · S(n,k) · y^k      (mod MOD)
        where S(n,k) is the Stirling number of the 2-nd kind.
        """
        # ------------------------------------------------------------------
        # factorials and inverse factorials (to get combinations quickly)
        # ------------------------------------------------------------------
        N = max(n, x)
        fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact = [1] * (N + 1)
        inv_fact[N] = pow(fact[N], MOD - 2, MOD)          # Fermat
        for i in range(N, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD

        # ------------------------------------------------------------------
        # Stirling numbers of the 2-nd kind – only final row (O(n²) time, O(n) mem)
        # recurrence:  S(n,k) = k·S(n-1,k) + S(n-1,k-1)
        # ------------------------------------------------------------------
        stirl = [0] * (n + 1)      # current row
        stirl[0] = 1               # S(0,0) = 1
        for i in range(1, n + 1):
            nxt = [0] * (n + 1)
            for k in range(1, i + 1):
                nxt[k] = (k * stirl[k] + stirl[k - 1]) % MOD
            stirl = nxt                     # advance to next row

        # ------------------------------------------------------------------
        # Accumulate the required sum
        #   term(k) = P(x,k) · S(n,k) · y^k,   where P(x,k)=x!/(x-k)!
        # ------------------------------------------------------------------
        ans = 0
        y_pow = 1
        max_k = min(n, x)
        for k in range(1, max_k + 1):
            y_pow = (y_pow * y) % MOD               # y^k
            perm = fact[x] * inv_fact[x - k] % MOD  # P(x,k)
            term = perm * stirl[k] % MOD
            term = term * y_pow % MOD
            ans = (ans + term) % MOD

        return ans