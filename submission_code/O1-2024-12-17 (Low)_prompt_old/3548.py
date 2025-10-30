class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        We say an integer x is k-palindromic if:
          1) x is a palindrome
          2) x is divisible by k

        An integer of length n (with no leading zero) is called 'good' if its digits can
        be rearranged to form a k-palindromic integer.

        We want to return how many n-digit integers (leading digit not zero) are good.

        Approach:
        1) We can generate all n-digit palindromes (leading digit not zero).
        2) Keep only those divisible by k.
        3) For each such palindrome, we store its digit multiset (digit counts from 0..9).
           - We store this as a tuple c0,c1,..,c9 in a set, so that each multiset is recorded only once.
        4) After collecting all multisets that can form a k-palindromic integer, we count how many distinct
           n-digit permutations (leading non-zero) each multiset has, and sum those counts.

        For n up to 10, enumerating all palindromes is feasible:
          - If n is even, let half_len = n//2. We iterate over all half_len-digit numbers with a non-zero first digit.
            Then mirror them to get an n-digit palindrome.
          - If n is odd, let half_len = (n+1)//2 (ceiling). Generate half_len-digit numbers (first digit not zero).
            Then mirror all but the middle digit to form an n-digit palindrome.
        We'll check divisibility by k, and if divisible, record its digit multiset.

        Finally, we compute:
            total permutations of a multiset of size n = n! / (c0! * c1! * ... * c9!)
            permutations that start with 0 (invalid) if c0 > 0
                = (n-1)! / ((c0-1)! * c1! * ... * c9!)
            So valid permutations that do not start with 0
                = total_perm - zero_leading_perm (if c0>0), else just total_perm.
        We sum that for each distinct multiset found.

        This yields the count of n-digit 'good' integers.
        """

        from math import factorial
        
        # Precompute factorials up to n=10
        fact = [factorial(i) for i in range(n+1)]
        
        def permutations_of_count(counts):
            """
            Given a list counts of length 10 representing the count of each digit (0..9),
            return the total number of permutations of those digits (multiset permutations).
            """
            total = sum(counts)
            res = fact[total]
            for c in counts:
                res //= fact[c]
            return res
        
        def leading_nonzero_permutations(counts):
            """
            Returns the number of permutations of the multiset of digits that do NOT start with zero.
            """
            total_perm = permutations_of_count(counts)
            if counts[0] == 0:
                # No zeros at all, or no digit-0 in the multiset => none start with 0
                return total_perm
            else:
                # Count how many permutations start with zero => remove them
                # That is permutations of the remaining n-1 digits with c0-1 zeros left
                counts[0] -= 1
                zero_leading = permutations_of_count(counts)
                counts[0] += 1
                return total_perm - zero_leading
        
        def digits_to_counts(s):
            """ Convert a string of digits into a tuple of digit counts. """
            c = [0]*10
            for ch in s:
                c[int(ch)] += 1
            return tuple(c)
        
        # Generate all n-digit palindromes, check divisibility by k, and collect their digit counts
        half_len = (n + 1) // 2  # Ceiling of n/2
        palindromic_sets = set()
        
        start = 10**(half_len - 1)  # smallest half_len-digit number (leading digit not zero)
        end = 10**half_len         # one past the largest half_len-digit number
        
        for half_num in range(start, end):
            half_str = str(half_num)
            if len(half_str) < half_len:
                # In theory half_str shouldn't be shorter, but in some edge cases, ensure correct length
                half_str = half_str.zfill(half_len)
            
            if n % 2 == 0:
                # Even length palindrome: mirror entire half
                pal_str = half_str + half_str[::-1]
            else:
                # Odd length palindrome: mirror all but the middle digit
                pal_str = half_str + half_str[-2::-1]
            
            pal_int = int(pal_str)
            if pal_int % k == 0:
                c_tuple = digits_to_counts(pal_str)
                palindromic_sets.add(c_tuple)
        
        # For each distinct digit multiset that can form a k-palindromic integer,
        # count the number of valid n-digit permutations of that multiset.
        result = 0
        for c_tuple in palindromic_sets:
            result += leading_nonzero_permutations(list(c_tuple))
        
        return result