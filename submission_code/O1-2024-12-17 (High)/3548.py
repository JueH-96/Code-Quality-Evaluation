class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        We will generate all n-digit palindromes (leading digit != 0) and check
        which are divisible by k. For each such palindrome, we record its digit
        multiset (the counts of each digit 0..9). Any n-digit integer having
        that same multiset of digits is "good" (since it can be rearranged to
        form a k-palindromic number). We then sum up the counts of all distinct
        n-digit permutations corresponding to those good multisets, taking care
        to exclude any permutations that start with 0 (leading zeros are disallowed).

        Steps:
        1. Enumerate all n-digit palindromes:
           - If n is even, let half_len = n//2.
             We iterate an integer 'half' from 10^(half_len-1) to 10^half_len - 1.
             The palindrome is formed by str(half) + reverse(str(half)).
           - If n is odd, let half_len = (n+1)//2.
             We iterate 'half' from 10^(half_len-1) to 10^half_len - 1.
             The palindrome is str(half) + reverse(str(half)[:-1]) (middle digit not duplicated).
           - Special case n=1 fits naturally with half_len=1, range(1..9).
        2. For each palindrome P (leading non-zero by construction), check if P % k == 0.
           If yes, compute the digit counts c0..c9 and store them in a set of "good" distributions.
        3. For each digit-distribution in "good" distributions, compute the number of valid n-digit
           permutations that do not start with 0:
              total_perm = n! / (c0! c1! ... c9!)
              leading_zero_perm = (n-1)! / ((c0-1)! c1! ... c9!) if c0 > 0 else 0
              valid_perm = total_perm - leading_zero_perm
           Sum these valid_perm values for all good distributions.
        4. Return the sum.
        This method is efficient for n <= 10 because the number of n-digit palindromes
        is at most 9 * 10^((n-1)//2) ≈ 90,000 for n=10, which is feasible.
        """

        import math

        # Precompute factorials up to n=10
        max_n = 10
        fact = [1] * (max_n+1)
        for i in range(1, max_n+1):
            fact[i] = fact[i-1] * i

        # A helper to compute the number of valid permutations (leading digit not zero)
        # for a given digit distribution c = [c0, c1, ..., c9].
        def valid_permutations(c):
            # total distinct permutations of n digits
            total_p = fact[n]
            for count in c:
                total_p //= fact[count]
            # subtract those permutations that start with 0 (if c0>0)
            if c[0] > 0:
                lead_zero_p = fact[n-1]
                # if we fix one digit '0' as the leading digit, we have c0-1 zeros left
                # and c1, c2, ... c9 unchanged for positions 2..n
                lead_zero_p //= fact[c[0]-1]
                for d in c[1:]:
                    lead_zero_p //= fact[d]
                return total_p - lead_zero_p
            else:
                return total_p

        good_distributions = set()

        # If n=1, half_len=1 => range(1..9)
        # For n>1 and even, half_len=n//2 => range(10^(half_len-1) .. 10^half_len -1)
        # For n>1 and odd, half_len=(n+1)//2 => range(10^(half_len-1) .. 10^half_len -1)
        if n == 1:
            start = 1
            end = 9
        else:
            if n % 2 == 0:
                half_len = n // 2
            else:
                half_len = (n + 1) // 2
            start = 10 ** (half_len - 1)
            end = 10 ** half_len - 1

        # Generate palindromes
        for half in range(start, end + 1):
            s = str(half)
            if n == 1:
                pal_str = s  # single-digit
            else:
                if n % 2 == 0:
                    # even length
                    pal_str = s + s[::-1]
                else:
                    # odd length
                    pal_str = s + s[:-1][::-1]

            pal_int = int(pal_str)
            # Check divisibility by k
            if pal_int % k == 0:
                # Build digit-count distribution
                c = [0]*10
                for ch in pal_str:
                    c[int(ch)] += 1
                good_distributions.add(tuple(c))

        # Sum valid permutations from all good distributions
        ans = 0
        for c_tuple in good_distributions:
            ans += valid_permutations(c_tuple)

        return ans