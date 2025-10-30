class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # The problem asks for the largest n-digit integer that is a palindrome and divisible by k.
        # We are looking for the largest number P such that:
        # 1. P has exactly n digits (no leading zeros).
        # 2. P is a palindrome.
        # 3. P % k == 0.

        # We will search for the largest palindrome by iterating through possible prefixes
        # in descending order and checking the conditions.

        # The number of digits in the left half (prefix) of an n-digit number.
        # For a palindrome, the second half is determined by the first half.
        # If n is odd, the middle digit belongs to the prefix.
        # If n is even, the prefix is exactly half the digits.
        # (n + 1) // 2 correctly gives ceil(n/2) for positive n.
        # E.g., n=1, half_len=1. n=2, half_len=1. n=3, half_len=2. n=4, half_len=2.
        half_len = (n + 1) // 2

        # Determine the range of prefix integers to check.
        # We want the largest palindrome, so we start with the largest possible prefix.
        # The largest integer with `half_len` digits is 10^half_len - 1 (e.g., for half_len=2, 10^2-1 = 99).
        start_prefix_int = 10**half_len - 1

        # The smallest integer with `half_len` digits is 10^(half_len - 1) (e.g., for half_len=2, 10^1 = 10).
        # This is the lower bound for our prefix search to ensure we are considering
        # prefixes that would form n-digit numbers when mirrored (or nearly mirrored).
        # For n=1, half_len=1, end_prefix_int = 10^(1-1) = 10^0 = 1. The smallest 1-digit prefix is 1.
        end_prefix_int = 10**(half_len - 1)

        # Iterate through all possible prefix integers in descending order.
        # These prefixes, when used to construct a palindrome, will give us
        # the n-digit palindromes in descending order.
        # We iterate from start_prefix_int down to end_prefix_int, inclusive.
        # The range(start, stop, step) function includes 'start' but excludes 'stop'.
        # To include end_prefix_int, the stop value must be end_prefix_int - 1.
        for prefix_int in range(start_prefix_int, end_prefix_int - 1, -1):
            # Convert the current prefix integer to its string representation.
            pref_str = str(prefix_int)

            # Construct the full palindrome string from the prefix string.
            # A palindrome is formed by the prefix followed by the reverse of its initial part.
            # The length of the prefix is half_len. The length of the full palindrome is n.
            # The reverse part is derived from the first n//2 characters of the prefix string reversed.
            # - If n is even (n=2m), half_len=m, n//2=m. The prefix is m digits. The reverse part is pref_str[:m], which is the entire prefix. pal = prefix + reverse(prefix).
            # - If n is odd (n=2m+1), half_len=m+1, n//2=m. The prefix is m+1 digits. The reverse part is pref_str[:m], which are the first m digits of the prefix. pal = prefix + (first m digits reversed).
            rev_half_str = pref_str[:n//2][::-1]
            pal_str = pref_str + rev_half_str

            # Check if the integer value of the palindrome string is divisible by k.
            # We use the property of modular arithmetic: (a * 10 + b) % k = ((a % k) * 10 + b) % k.
            # This allows us to calculate the remainder iteratively digit by digit.
            # This is necessary for large n, as converting the entire pal_str to a standard
            # integer type could exceed memory limits or be slow. Python's arbitrary
            # precision integers handle large numbers, but this iterative method is explicit
            # and guarantees an O(n) operation for the divisibility check per palindrome.
            remainder = 0
            for digit_char in pal_str:
                # Update the remainder: current_remainder = (previous_remainder * 10 + current_digit) % k
                remainder = (remainder * 10 + int(digit_char)) % k

            # If the final remainder is 0, the palindrome is divisible by k.
            # Since we are iterating prefixes downwards from the largest possible one,
            # the first palindrome we find that meets the divisibility condition
            # is guaranteed to be the largest k-palindromic number with n digits.
            if remainder == 0:
                return pal_str

        # This return statement should ideally not be reached given the problem constraints.
        # For any n >= 1 and k >= 1, there should exist an n-digit k-palindromic number.
        # The largest such number must be constructed from a prefix of length half_len
        # if it exists among the largest palindromes. This search strategy covers
        # all such palindromes starting from the largest.
        # It's included as a fallback, though typically not expected to be hit in tests.
        return "-1" # Should not happen in a valid test case according to problem type.