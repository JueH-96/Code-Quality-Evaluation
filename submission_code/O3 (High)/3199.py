from math import comb

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        """
        Count the number of 3–tuples (a, b, c) of non-negative integers such that
            a + b + c = n   and   0 ≤ a, b, c ≤ limit
        using an inclusion–exclusion formula.

        The number of non-negative solutions of a + b + c = n is C(n+2, 2).
        Let Ei be the set of solutions where child i gets more than `limit`
        candies.  Re-writing x_i = y_i + limit + 1 (y_i ≥ 0) shows
            |Ei| = C(n-(limit+1)+2, 2) when n ≥ limit+1.
        Inclusion–exclusion over the three children gives the final answer.
        """
        # If the total number of candies exceeds the maximum each child can get,
        # distribution is impossible.
        if n > 3 * limit:
            return 0

        answer = 0
        shift = limit + 1           # one "forbidden" candy per child
        for k in range(4):          # number of children that break the limit
            remaining = n - k * shift
            if remaining < 0:
                continue            # this intersection is empty
            ways = comb(remaining + 2, 2)  # unconstrained solutions for rest
            term = comb(3, k) * ways       # choose which k children exceed
            answer += term if k % 2 == 0 else -term  # inclusion–exclusion

        return answer