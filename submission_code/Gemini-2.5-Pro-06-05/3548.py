import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        """
        Counts the number of n-digit "good" integers.

        A number is "good" if its digits can be rearranged to form a k-palindromic number.
        A k-palindromic number is a palindrome that is divisible by k.
        The rearranged number must also have n digits (no leading zeros).
        """
        
        # Precompute factorials for permutation calculations.
        factorials = [math.factorial(i) for i in range(n + 1)]

        def permutations_count(counts_tuple: tuple) -> int:
            """
            Calculates the number of valid n-digit integers that can be formed from
            the multiset of digits. A number is valid if it doesn't have a leading zero.
            """
            denominator = 1
            for count in counts_tuple:
                denominator *= factorials[count]

            c0 = counts_tuple[0]  # Count of the digit '0'
            
            # If all digits are not zero, all permutations are valid n-digit numbers.
            # Otherwise, we calculate valid permutations by ensuring the first digit is non-zero.
            # The formula (total_permutations * (n - c0) / n) simplifies to:
            # ((n-1)! * (n - c0)) / (c0! * c1! * ... * c9!)
            if n > c0:
                return (factorials[n - 1] * (n - c0)) // denominator
            else: # n == c0, meaning all digits are 0.
                return 0

        # Step 1: Find all unique multisets of digits that can form a k-palindromic number.
        good_digit_counts = set()
        
        m = (n + 1) // 2
        # The first half of an n-digit palindrome is an m-digit number (or 1-digit for n=1).
        # It must not start with 0.
        start = 10**(m - 1)
        end = 10**m
        
        # Iterate through all possible first halves to generate all palindromes.
        for first_half_num in range(start, end):
            first_half_str = str(first_half_num)
            
            # Construct the full palindrome string.
            if n % 2 == 0:
                # e.g., n=4, first_half="12" -> palindrome="1221"
                palindrome_str = first_half_str + first_half_str[::-1]
            else:
                # e.g., n=5, first_half="123" -> palindrome="12321"
                second_half_str = first_half_str[:-1][::-1]
                palindrome_str = first_half_str + second_half_str
            
            palindrome_num = int(palindrome_str)
            
            # Check if it's k-palindromic.
            if palindrome_num % k == 0:
                # If so, record its digit multiset.
                counts = [0] * 10
                for digit in palindrome_str:
                    counts[int(digit)] += 1
                good_digit_counts.add(tuple(counts))

        # Step 2: Sum the permutation counts for each unique good digit multiset.
        total_good_integers = 0
        for counts in good_digit_counts:
            total_good_integers += permutations_count(counts)
            
        return total_good_integers