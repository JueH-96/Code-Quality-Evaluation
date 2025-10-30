import math

class Solution:
    def waysToReachStair(self, k: int) -> int:
        """
        Let n be the number of ‘up’ operations that are used.
        jump starts from 0, hence the lengths of the n up-jumps are
        1, 2, 4, … , 2**(n-1).  
        Their total height is (2**n – 1).

        If d is the number of ‘down’ ( −1 ) moves, the final position is

            1 + (2**n – 1) – d  =  2**n – d .

        To finish on stair k we need
                  d = 2**n – k ,  d ≥ 0.

        While executing the moves two extra rules must hold
          • no two downs are consecutive
          • we never try to go down from stair 0.

        Thanks to the “no consecutive downs’’ rule, once we step from 1 to 0
        we must go up next, so the second rule is automatically satisfied.
        Therefore a valid sequence is simply an arrangement of

                 n  ‘U’s   and   d  ‘D’s

        with no two ‘D’s adjacent.  If we fix n (hence d), the n ‘U’s create
        (n+1) gaps ( _ U _ U … U _ ).  Choosing any d of these gaps and
        placing one ‘D’ in each gives every admissible sequence once, hence

                 number_of_sequences = C(n+1, d).

        Conditions that must be met for given n:
            d = 2**n – k  must satisfy 0 ≤ d ≤ n+1.

        k ≤ 10^9 → n never exceeds 31 (2**30 ≈ 1.07·10^9),
        so we can simply iterate over n = 0 … 31 and sum the
        valid combinations.
        """

        ans = 0
        # 2**31 > 1e9, that is enough for all k ≤ 1e9
        for n in range(32):
            d = (1 << n) - k          # 2**n - k
            if 0 <= d <= n + 1:       # must be non-negative and fit in gaps
                ans += math.comb(n + 1, d)
        return ans