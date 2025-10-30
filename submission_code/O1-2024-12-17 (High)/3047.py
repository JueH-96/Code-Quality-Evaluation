class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        """
        ----------------------------------------------------------------------
        Interpretation (based on the examples given):
        
        We have an array nums of length n (1-indexed). We want to choose a
        subset of the index set {1, 2, ..., n} such that for every pair of
        chosen indices (i, j), the product i*j is a perfect square. Among all
        such subsets, we want to maximize the sum of the corresponding values
        nums[i].

        Why indices rather than values?
        --------------------------------
        The provided examples only make sense if "complete subset" refers to
        the *indices* themselves being pairwise perfect-square when multiplied.
        For instance, in Example 1:

            nums = [8,7,3,5,7,2,4,9], n=8

            The example states that {1,4} (→ product of indices 1*4=4) and
            {2,8} (→ product 2*8=16) are both “complete” subsets of indices.
            Checking their sums of nums-values:
              - Indices {1,4} → 8 + 5 = 13
              - Indices {2,8} → 7 + 9 = 16 (maximum)

            That only holds if we interpret "i*j is a perfect square" for
            (i, j) in the chosen index subset.

        Likewise, in Example 2:

            nums = [5,10,3,10,1,13,7,9,4], n=9

            The subsets mentioned (e.g. {1,4}, {1,9}, {2,8}, {4,9}, {1,4,9})
            become “complete” precisely because the product of those indices
            is a perfect square for every pair:
              - 1*4 = 4  (square),
              - 1*9 = 9  (square),
              - 2*8 = 16 (square),
              - 4*9 = 36 (square),
              - and so on.

            They then compute sums of nums[...] from those indices.

        Therefore, the real condition is:
          "Pick a subset of {1..n} so that for any i, j in that subset,
           i*j is a perfect square."

        Mathematical restatement:
        -------------------------
        The condition i*j = perfect square  ⇔  the parity (mod 2) of
        the prime-factor-exponents of i and j are the same.  In other words,
        two indices i and j can be in the same “complete” set if and only if
        they share the same prime-factorization-parity-vector.

        Hence, to form a valid subset, *all* chosen indices must share the
        same prime-factor parity signature.

        Algorithm:
        ----------
         1) For each index i in [1..n], factor i and record which primes appear
            with odd exponent.  Store those primes in a canonical form (e.g.,
            a sorted tuple of prime bases).
         2) Group indices by that same tuple.  Because within the same group,
            every pair (i, j) has the same parity-vector and thus i*j is
            a perfect square.
         3) For each group, sum up nums[i-1] (since i is 1-based).  The entire
            group is valid, so we take all of them to maximize the sum.
         4) Return the maximum such group sum.

        Complexity:
        -----------
        - n <= 10^4
        - We factor each i up to n (worst-case 10^4).  Each factorization is
          up to sqrt(i) ~ 100 operations → ~ 10^6 total, which is acceptable
          in optimized Python.
        - We then aggregate sums in a dictionary keyed by the prime-parity
          tuple, and pick the maximum.

        This matches the examples exactly.
        ----------------------------------------------------------------------
        """

        import math
        from collections import defaultdict

        # Helper: factor i into primes with odd exponent, return as sorted tuple
        def factor_odd_primes(x: int) -> tuple:
            if x == 1:
                return ()
            # We'll flip membership in a set for each prime factor.
            # In the end, only primes with odd exponent remain in the set.
            odd_factors = set()
            
            # Factor out 2
            while x % 2 == 0:
                if 2 in odd_factors:
                    odd_factors.remove(2)
                else:
                    odd_factors.add(2)
                x //= 2
            
            # Factor out odd primes
            f = 3
            while f * f <= x:
                while x % f == 0:
                    if f in odd_factors:
                        odd_factors.remove(f)
                    else:
                        odd_factors.add(f)
                    x //= f
                f += 2
            
            # If remainder > 1, it's a prime
            if x > 1:
                if x in odd_factors:
                    odd_factors.remove(x)
                else:
                    odd_factors.add(x)
            
            # Return canonical sorted tuple
            return tuple(sorted(odd_factors))

        # Dictionary: parity_vector -> sum of nums for indices having that vector
        group_sum = defaultdict(int)

        n = len(nums)
        for i in range(1, n+1):
            sig = factor_odd_primes(i)
            group_sum[sig] += nums[i-1]

        # The answer is the maximum sum over all parity-vector groups
        return max(group_sum.values())