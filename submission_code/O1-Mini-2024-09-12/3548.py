class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from itertools import product
        from math import factorial
        from collections import defaultdict

        # Precompute factorials up to 10!
        factorials = [1] * (n+1)
        for i in range(1, n+1):
            factorials[i] = factorials[i-1] * i

        # Function to compute digit counts as tuple
        def get_digit_count(number):
            counts = [0]*10
            for ch in str(number):
                counts[int(ch)] +=1
            return tuple(counts)

        # Generate all n-digit palindromes
        half_length = (n +1)//2
        start = 10**(half_length-1)
        end = 10**half_length
        valid_counts = set()

        for half in range(start, end):
            half_str = str(half)
            if n %2 ==0:
                full_str = half_str + half_str[::-1]
            else:
                full_str = half_str + half_str[:-1][::-1]
            palindrome = int(full_str)
            if palindrome %k ==0:
                digit_count = get_digit_count(palindrome)
                valid_counts.add(digit_count)

        # If no valid palindromic digit counts
        if not valid_counts:
            return 0

        # Function to compute the number of n-digit numbers with given digit count
        def count_numbers(counts):
            total = factorials[n]
            for cnt in counts:
                total //= factorials[cnt]
            if counts[0] >0:
                # Subtract numbers with leading zero
                total_zero = factorials[n-1]
                counts_zero = list(counts)
                counts_zero[0] -=1
                for cnt in counts_zero:
                    total_zero //= factorials[cnt]
                total -= total_zero
            return total

        # Use a set to avoid duplicate digit counts
        unique_counts = valid_counts

        # Calculate the total number of good integers
        result = 0
        for counts in unique_counts:
            result += count_numbers(counts)

        return result