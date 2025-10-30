from typing import List

MOD = 1_000_000_007


class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        m = n - len(sick)                # number of children that are healthy
        if m == 0:                       # all children are already sick
            return 1

        # ----------------- factorials and inverse factorials -----------------
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD
        # --------------------------------------------------------------------

        #   answer starts with  m!  (all ways to interleave the segments)
        ans = fact[m]

        #  power of two exponent contributed by the internal gaps
        pow_two_exp = 0

        # ------------ deal with the leftmost and rightmost gaps -------------
        left_gap = sick[0]                 # positions 0 … sick[0]-1
        ans = ans * inv_fact[left_gap] % MOD

        right_gap = (n - 1) - sick[-1]     # positions sick[-1]+1 … n-1
        ans = ans * inv_fact[right_gap] % MOD
        # --------------------------------------------------------------------

        # --------------------- internal gaps ---------------------------------
        for i in range(len(sick) - 1):
            gap = sick[i + 1] - sick[i] - 1     # length of the gap
            ans = ans * inv_fact[gap] % MOD     # denominator of multinomial

            if gap:                             # only real gaps contribute
                pow_two_exp += gap - 1          # 2^(gap-1) possibilities
        # --------------------------------------------------------------------

        # multiply the contributions of all internal gaps (2^(Σ(gap-1)))
        ans = ans * pow(2, pow_two_exp, MOD) % MOD
        return ans