class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        We say an integer x is k-palindromic if:
         1) x is a palindrome (leading digit != 0),
         2) x is divisible by k.

        An integer is 'good' if its digits can be rearranged to form a k-palindromic integer.
        We return the count of all n-digit good integers (with no leading zeros).

        Approach:
         - Generate all n-digit palindromes (leading digit != 0).
         - Check which are divisible by k.
         - For each valid palindrome, record its digit multiset (the counts of each digit 0..9).
           That means any n-digit number with exactly this multiset of digits is 'good' because
           those digits can be rearranged to form the valid palindrome.
         - To finish, for each valid digit multiset we add up how many distinct n-digit permutations
           it can form (i.e. leading digit != 0). Summing these counts gives the answer.

        Because n <= 10 and k <= 9, generating all n-digit palindromes (up to 90,000 for n=10) is feasible
        in Python. We then store each palindrome's digit counts in a set (to avoid duplicates).
        Finally we compute, for each distinct digit multiset, how many n-digit permutations (no leading zero)
        it can form, and sum these counts.
        """

        # Precompute factorials up to 10 for permutation counts.
        fact = [1] * 11
        for i in range(1, 11):
            fact[i] = fact[i - 1] * i

        def digit_count(num_str):
            """Return a tuple of length 10 with counts of each digit in num_str."""
            counts = [0]*10
            for ch in num_str:
                counts[ord(ch) - ord('0')] += 1
            return tuple(counts)

        def count_permutations_no_leading_zero(counts):
            """
            Given a digit-count tuple (counts of 0..9 in the number),
            total distinct permutations of length n = n! / (counts[0]! * counts[1]! * ... * counts[9]!).
            We must subtract those that start with digit '0' if 0 is present.
            """
            total_digits = sum(counts)
            # Compute total permutations
            total_perm = fact[total_digits]
            for c in counts:
                total_perm //= fact[c]

            # Compute how many permutations lead with zero (if zero is present).
            zero_perm = 0
            if counts[0] > 0:
                zero_perm = fact[total_digits - 1]
                # if we fix a '0' in the first position, we have (counts[0] - 1) zeroes left
                # plus the unchanged counts of the other digits.
                for d in range(10):
                    if d == 0:
                        zero_perm //= fact[counts[0] - 1]
                    else:
                        zero_perm //= fact[counts[d]]

            return total_perm - zero_perm

        # Edge case: if n=1, just check digits 1..9 quickly:
        # (this will also be covered by the palindrome-generation logic below,
        #  but we include a small check for clarity/confirmation)
        if n == 1:
            # digits 1..9, check how many are divisible by k
            ans = 0
            for d in range(1, 10):
                if d % k == 0:
                    ans += 1
            return ans

        # Generate palindromes of length n
        # If n is even, half_len = n//2
        # If n is odd,  half_len = n//2 + 1
        # Then range from 10^(half_len-1) to 10^half_len - 1 ensures
        # the first digit won't be zero (leading to an n-digit palindrome).
        if n % 2 == 0:
            half_len = n // 2
            odd = False
        else:
            half_len = n // 2 + 1
            odd = True

        start = 10 ** (half_len - 1)
        end = 10 ** half_len

        valid_multisets = set()  # store digit-count tuples that yield a palindrome divisible by k

        for front in range(start, end):
            s_front = str(front)
            if odd:
                # skip the last digit of s_front for the mirrored part
                pal_str = s_front + s_front[:-1][::-1]
            else:
                pal_str = s_front + s_front[::-1]

            # Now check divisibility
            # (pal_str can be up to 10 digits, well within normal Python int range).
            pal_int = int(pal_str)
            if pal_int % k == 0:
                # Record its digit multiset
                dcount = digit_count(pal_str)
                valid_multisets.add(dcount)

        # For each valid digit multiset, add up permutations that yield an n-digit number
        ans = 0
        for dcount in valid_multisets:
            ans += count_permutations_no_leading_zero(dcount)

        return ans