class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        import math
        from collections import defaultdict

        # Factorize a number modulo 2 for each prime factor's exponent.
        # Return a frozenset of primes that appear an odd number of times in the factorization.
        def factorize_mod2(x: int) -> frozenset:
            factors = set()
            # Handle factor 2 separately
            while x % 2 == 0:
                if 2 in factors:
                    factors.remove(2)
                else:
                    factors.add(2)
                x //= 2

            # Trial division by odd integers
            f = 3
            while f * f <= x:
                while x % f == 0:
                    if f in factors:
                        factors.remove(f)
                    else:
                        factors.add(f)
                    x //= f
                f += 2

            # If there's anything left > 1, it's prime
            if x > 1:
                if x in factors:
                    factors.remove(x)
                else:
                    factors.add(x)

            return frozenset(factors)

        # Group numbers by their factorization mod 2 signature
        group_sums = defaultdict(int)
        for num in nums:
            signature = factorize_mod2(num)
            group_sums[signature] += num

        # The answer is the maximum sum among all groups
        return max(group_sums.values())