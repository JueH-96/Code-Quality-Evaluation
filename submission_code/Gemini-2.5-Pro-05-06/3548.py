import math

class Solution:
  
    def _calculate_permutations(self, counts_tuple: tuple[int, ...], n: int, fact: list[int]) -> int:
        """
        Calculates the number of n-digit integers that can be formed from the given
        digit counts, without leading zeros.
        counts_tuple: tuple of 10 integers, (count_of_0, count_of_1, ..., count_of_9).
        n: number of digits.
        fact: precomputed factorials.
        """
        c0 = counts_tuple[0] # Count of digit '0'
        
        denominator = 1
        for count_val in counts_tuple:
            denominator *= fact[count_val]
        
        # This case should not be hit if inputs are valid and fact[0]=1.
        if denominator == 0: 
            return 0

        # Total permutations, including those that might start with a zero.
        P_total = fact[n] // denominator
        
        if c0 == 0:
            # If there are no zeros, no permutation can start with zero.
            return P_total
        else:
            # If there are zeros, subtract permutations starting with '0'.
            # Number of permutations starting with '0' = P_total * (c0 / n).
            # Valid permutations = P_total * (1 - c0 / n) = P_total * (n - c0) / n.
            # P_total * c0 is guaranteed to be divisible by n, so P_total * (n-c0) is also divisible by n.
            P_valid = (P_total * (n - c0)) // n
            return P_valid

    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials from 0! to n!
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i
            
        # half_len is the number of digits in the first half of the palindrome
        half_len = (n + 1) // 2 # Equivalent to math.ceil(n / 2)
        
        # Determine the range for h_val, which represents the integer formed by the first half digits.
        if half_len == 1: # For n=1 or n=2. h_val is a single digit (1-9).
            start_h_val = 1
        else: # For n > 2.
            start_h_val = 10**(half_len - 1)
        
        end_h_val = 10**half_len # range is exclusive for the end value (h_val < end_h_val)

        found_digit_counts = set() # To store unique multisets of digits from k-palindromic numbers

        # Optimization: Prune search space for h_val.
        # The first digit of h_val is the first digit (d1) of the palindrome.
        # d1 is also the last digit of the palindrome.
        d1_allowed_by_k = list(range(1, 10)) # d1 must be non-zero.

        if k % 2 == 0: # If k is even, d1 (last digit) must be even.
            d1_allowed_by_k = [d for d in d1_allowed_by_k if d % 2 == 0]
        
        if k % 5 == 0: # If k is a multiple of 5, d1 (last digit) must be 5 (cannot be 0).
            d1_allowed_by_k = [d for d in d1_allowed_by_k if d == 5]
        
        for h_val in range(start_h_val, end_h_val):
            s_h = str(h_val) # String representation of the first half
            
            # Apply d1 optimization filter
            first_digit_of_h = int(s_h[0])
            if first_digit_of_h not in d1_allowed_by_k:
                continue

            # Construct the palindrome string from s_h
            s_palindrome: str
            if n % 2 == 1: # Odd length palindrome
                # e.g., n=1, s_h="d", s_h[:-1]="", s_pal="d"
                # e.g., n=3, s_h="d1d2", s_h[:-1]="d1", s_pal="d1d2" + reverse("d1") = "d1d2d1"
                s_palindrome = s_h + s_h[:-1][::-1]
            else: # Even length palindrome
                # e.g., n=2, s_h="d1", s_pal="d1" + reverse("d1") = "d1d1"
                s_palindrome = s_h + s_h[::-1]
            
            p_val = int(s_palindrome) # Integer value of the palindrome
            
            if p_val % k == 0:
                # p_val is k-palindromic. Get its digit multiset.
                counts = [0] * 10 # counts[i] is frequency of digit i
                for digit_char in s_palindrome:
                    counts[int(digit_char)] += 1
                found_digit_counts.add(tuple(counts))
        
        # Sum permutations for all unique valid digit multisets found
        total_good_integers = 0
        for counts_tuple in found_digit_counts:
            total_good_integers += self._calculate_permutations(counts_tuple, n, fact)
            
        return total_good_integers