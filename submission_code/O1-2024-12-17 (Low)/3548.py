class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        We say that an n-digit integer x (leading digit != 0) is 'good' if its digits
        can be rearranged to form a palindrome which is divisible by k.

        We want to count how many distinct n-digit integers are good.

        Approach:
        1) Generate all n-digit palindromes with leading digit != '0'.
        2) Check which palindromes are divisible by k.
        3) For each palindrome that is divisible by k, collect its multiset of digits
           (i.e., sorted digits).
        4) For each distinct digit-multiset that occurs among such palindromes,
           count how many n-digit permutations (leading digit != 0) that multiset produces.
        5) Sum these counts to get the final answer.

        Since n <= 10, generating all n-digit palindromes is feasible:
          - For n even, say n=10 => half-length = 5 => we choose the first half from [10^(4) .. 10^5 - 1].
          - For n odd, say n=9  => half-length = 5 => we choose from [10^(4) .. 10^5 - 1],
            then reflect all but the middle digit to get the second half.
          - This is at most around 90,000 palindromes to check for n=10, which is manageable.

        We'll use a set (or dict) to track which digit-multisets are "good." Then, at the end,
        for each good multiset, we add up the number of distinct permutations of size n
        that do not begin with '0'. That total is our result.
        """

        from math import factorial

        # Precompute factorials up to 10 for permutation counting
        fact = [1]*(n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1]*i

        def permutations_without_leading_zero(counts):
            """
            Given counts of digits (counts[0..9]) summing to n,
            return the number of distinct permutations of length n
            whose leading digit is not zero.
            """
            # total permutations = n! / (c0! * c1! * ... * c9!)
            total = fact[n]
            for d in counts:
                total //= fact[d]

            # subtract permutations that start with '0' (if possible)
            if counts[0] > 0:
                # fix '0' in front => now we have n-1 positions left
                # and counts[0] - 1 zeros remain
                bad = fact[n-1]
                for idx, c in enumerate(counts):
                    if idx == 0:
                        bad //= fact[c-1]
                    else:
                        bad //= fact[c]
                return total - bad
            else:
                return total

        def digit_multiset(num):
            """
            Return a tuple of counts of each digit 0..9 in the integer num.
            """
            s = str(num)
            c = [0]*10
            for ch in s:
                c[ord(ch)-ord('0')] += 1
            return tuple(c)

        # Special case: if n=1, just check digits 1..9
        if n == 1:
            # Count how many single digits 1..9 are divisible by k
            cnt = 0
            for d in range(1,10):
                if d % k == 0:
                    cnt += 1
            return cnt

        # Generate all n-digit palindromes with leading digit != '0'
        # We'll store the digit-multisets of those palindromes that are divisible by k.
        good_multisets = set()

        half_len = (n+1)//2  # for both even and odd n, this covers the "top half" (including middle for odd n)

        start = 10**(half_len-1)  # e.g. if half_len=2, start=10; if half_len=3, start=100
        end = 10**half_len        # one past the last number to consider

        # Iterate all possible "first halves"
        for half in range(start, end):
            s = str(half)
            if len(s) < half_len:
                # leading zeros (we skip, because that wouldn't yield an n-digit palindrome)
                continue

            if n % 2 == 0:
                # even-length palindrome: mirror the entire half
                pal_str = s + s[::-1]
            else:
                # odd-length palindrome: mirror all but the last digit of s
                pal_str = s + s[-2::-1]

            # Convert to int
            pal_num = int(pal_str)
            # Check divisibility
            if pal_num % k == 0:
                # If divisible, record its digit multiset
                dset = digit_multiset(pal_num)
                good_multisets.add(dset)

        # Now compute how many n-digit numbers each "good" multiset can form (leading digit != 0)
        ans = 0
        for dset in good_multisets:
            ans += permutations_without_leading_zero(dset)

        return ans