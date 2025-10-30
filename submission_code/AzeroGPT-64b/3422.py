class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 1000000007
        if k == 0:
            return 1
        if n == 1:
            return 1
        if n > k:
            return Solution.valueAfterKSeconds(n - 1, k) * n % mod
        from math import comb
        if n < k + 3:
            return comb(k + 1, n - 1) * pow(2, n - 1, mod) % mod
        # for large n, n choose (n-1), multiplied by 2^(n-1).
        # The denominator is (n-1)!, which can be cancelled with the (2k choose (n-1))! in the numerator.
        # So the result can be simplified to (2k choose (n-1))/k.
        k_factorial_inverse = pow(comb(2 * k, n - 1), mod - 2, mod)
        return comb(2 * k, n - 1) * pow(2, n - 1, mod) * k_factorial_inverse % mod