MOD = 10 ** 9 + 7

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        """
        Let N = m*n be the amount of cells on the board.
        For every unordered pair of different cells (c1, c2)
        its Manhattan distance contributes to the answer in every
        arrangement that contains both cells.  
        The amount of such arrangements is C(N-2, k-2)
        (choose the remaining k-2 cells among the other N-2).

        Therefore  
            answer = C(N-2, k-2) * (sum of Manhattan distances of all pairs)

        The pair-distance sum factorises into a “row part”
        and a “column part”:

            Σ|x1-x2|   over all pairs = n² · Σ_{d=1}^{m-1} d·(m-d)
            Σ|y1-y2|   over all pairs = m² · Σ_{d=1}^{n-1} d·(n-d)

        The inner sum is the same closed form on both axes:
            Σ_{d=1}^{L-1} d·(L-d) = L(L-1)(L+1)/6

        Putting everything together gives
            S = n²·m(m-1)(m+1)/6 + m²·n(n-1)(n+1)/6

        Finally the result is  ( S mod MOD ) · C(N-2,k-2) mod MOD.
        Because N ≤ 10⁵, factorials up to N are enough to compute
        the single binomial coefficient.
        """
        N = m * n                                   # total cells
        
        # -------- pre–compute factorials up to N (N ≤ 1e5) -------------
        fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        inv_fact = [1] * (N + 1)
        inv_fact[N] = pow(fact[N], MOD - 2, MOD)    # Fermat inversion
        for i in range(N, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD
        
        def comb(a: int, b: int) -> int:
            if b < 0 or b > a:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD
        
        # ------------ coefficient: in how many subsets a pair appears ------------
        pair_factor = comb(N - 2, k - 2)            # C(N-2, k-2)  (k ≥ 2)
        
        # ------------ sum of distances of all unordered pairs ------------
        # row contribution
        SR = m * (m - 1) * (m + 1) // 6             # Σ_{d} d·(m-d)
        # column contribution
        SC = n * (n - 1) * (n + 1) // 6
        
        # total sum S   (use modular reduction only at the very end)
        S_mod = ( (n * n % MOD) * (SR % MOD) + (m * m % MOD) * (SC % MOD) ) % MOD
        
        # ---------------- final result -----------------
        return S_mod * pair_factor % MOD