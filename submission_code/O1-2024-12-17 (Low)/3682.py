class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        """
        We want the number of length-n arrays with elements in [1, m] such that
        exactly k adjacent pairs are equal.

        A well-known combinatorial argument shows that the number of such arrays is:
            C(n-1, k) * m * (m-1)^(n-1-k)
        modulo 1e9+7.

        Explanation:
        - Let "runs" be blocks of consecutive identical elements in the array.
          If an array has r runs, then there are (r-1) positions where adjacent
          elements differ and thus (n-1) - (r-1) = n-r positions where adjacent
          elements are equal.
        - To have exactly k equal-adjacent pairs, we need k = (n-r).
          So r = n - k.
        - The ways to pick which indices define the r runs (all runs have at least
          one element) is C(n-1, r-1). This partitions the n elements into r runs.
        - For each run, we pick values from [1, m] such that adjacent runs use
          different values. The first run can use any of m possibilities, and each
          of the remaining (r-1) runs can use any of (m-1) possibilities (since it
          must differ from the previous run), giving m * (m-1)^(r-1).
        - Substituting r = n - k, the final formula becomes:
            C(n-1, k) * m * (m-1)^(n-1-k).

        We'll implement this with precomputed factorials and modular inverses
        to efficiently compute C(n-1, k) under modulo 1e9+7.
        """

        MOD = 10**9 + 7

        # Special quick cases
        if n == 1:
            # If n=1, there's no pair. Only valid if k=0. In that case, answer = m.
            return m % MOD if k == 0 else 0

        # Precompute factorials and inverse factorials for combination.
        # We only need up to n, which is at most 10^5.
        max_val = n  # a bit above n-1 if you prefer
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)

        for i in range(2, max_val + 1):
            fact[i] = fact[i-1] * i % MOD

        # Fermat's little theorem for inverse: a^(MOD-2) ≡ a^(-1) (mod MOD)
        inv_fact[max_val] = pow(fact[max_val], MOD-2, MOD)
        for i in reversed(range(max_val)):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

        def comb(n_, r_):
            if r_ < 0 or r_ > n_:
                return 0
            return fact[n_] * inv_fact[r_] % MOD * inv_fact[n_-r_] % MOD

        # Compute the result using the formula: C(n-1, k) * m * (m-1)^(n-1-k)
        c = comb(n-1, k)
        ways_m1 = pow(m-1, (n-1-k), MOD)
        ans = (c * m) % MOD
        ans = (ans * ways_m1) % MOD
        return ans