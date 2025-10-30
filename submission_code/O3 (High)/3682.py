MOD = 10 ** 9 + 7

class Solution:
    # ----------  tools ----------
    @staticmethod
    def _build_factorials(limit: int):
        """
        Pre–compute factorials and inverse factorials modulo MOD
        up to `limit'.
        """
        fact = [1] * (limit + 1)
        for i in range(1, limit + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact = [1] * (limit + 1)
        inv_fact[limit] = pow(fact[limit], MOD - 2, MOD)          # Fermat
        for i in range(limit - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        return fact, inv_fact

    @staticmethod
    def _nCk(n: int, k: int, fact, inv_fact) -> int:
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
    # --------------------------------

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        """
        #segments  s = n - k
        answer  = C(n-1, s-1) * m * (m-1)^(s-1)   (mod MOD)
        """
        s = n - k                                  # number of equal–value segments
        if s <= 0:                                 # impossible, but guarded anyway
            return 0

        # Pre–compute factorials up to n because we need C(n-1, s-1)
        fact, inv_fact = self._build_factorials(n)

        ways_lengths = self._nCk(n - 1, s - 1, fact, inv_fact)     # choose segment cuts
        ways_values  = m * pow(m - 1, s - 1, MOD) % MOD            # color the segments

        return ways_lengths * ways_values % MOD