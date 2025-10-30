from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # --------------------------------------------------------------------
        # A subset is "complete" if the product of every pair of its elements
        # is a perfect square. One can show that this happens exactly when
        # all elements in the subset share the same prime-factor-parity pattern.
        #
        # In other words, for each number we determine which primes appear
        # with an odd exponent in its factorization (the "square-free kernel").
        # All numbers having the same "square-free kernel" can form a complete
        # subset. Thus, we group numbers by that kernel, sum each group, and
        # the largest group-sum is our answer.
        # --------------------------------------------------------------------
        
        # Step 1: Precompute primes up to 31622 (sqrt(10^9)) via sieve.
        max_p = 31622
        primes = self._sieve_primes(max_p)

        # Step 2: Dictionary mapping frozenset-of-prime-factors (odd exponent)
        #         to the sum of the numbers that share that factor parity.
        group_sums = {}

        # Step 3: Process each number, factor it, compute its square-free kernel,
        #         and accumulate its value in the corresponding group.
        for x in nums:
            kernel = self._factor_parity(x, primes)
            group_sums[kernel] = group_sums.get(kernel, 0) + x
        
        # Step 4: The result is the maximum sum among all groups.
        return max(group_sums.values())

    def _sieve_primes(self, n: int) -> List[int]:
        """ Return a list of primes up to n (inclusive) using a simple sieve. """
        sieve = [True] * (n + 1)
        sieve[0], sieve[1] = False, False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    def _factor_parity(self, x: int, primes: List[int]) -> frozenset:
        """
        Factorize x with the given list of primes, and collect all primes
        with an odd exponent. Return a frozenset of these primes.
        """
        factors_odd = set()
        temp = x

        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    # Toggle p (i.e. add if not in set, remove if it was added)
                    if p in factors_odd:
                        factors_odd.remove(p)
                    else:
                        factors_odd.add(p)
        # If there's anything left > 1, it must be prime
        if temp > 1:
            p = temp
            if p in factors_odd:
                factors_odd.remove(p)
            else:
                factors_odd.add(p)

        return frozenset(factors_odd)