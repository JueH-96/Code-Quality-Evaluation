import math
from collections import Counter

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to 10!
        factorial = [1] * (n+1)
        for i in range(1, n+1):
            factorial[i] = factorial[i-1] * i
        
        # Function to generate all n-digit palindromic numbers
        def generate_palindromes(n):
            half = (n + 1) // 2
            start = 10**(half - 1) if half > 1 else 1
            end = 10**half
            for first_half in range(start, end):
                first_half_str = str(first_half)
                if n % 2 == 0:
                    palindrome_str = first_half_str + first_half_str[::-1]
                else:
                    palindrome_str = first_half_str + first_half_str[:-1][::-1]
                palindrome = int(palindrome_str)
                yield palindrome
        
        unique_freq = set()
        total_count = 0
        
        # Generate all n-digit palindromes divisible by k
        for palindrome in generate_palindromes(n):
            if palindrome % k != 0:
                continue
            # Get frequency distribution
            freq = Counter(str(palindrome))
            freq_tuple = tuple(freq.get(str(digit), 0) for digit in range(10))
            if freq_tuple in unique_freq:
                continue
            unique_freq.add(freq_tuple)
            # Calculate total permutations
            total_freq = n
            denom = 1
            for count in freq.values():
                denom *= math.factorial(count)
            total_perms = math.factorial(total_freq) // denom
            # Calculate permutations with zero first
            if freq.get('0', 0) > 0:
                # Fix zero in first position, reduce zero count by 1
                freq_zero = freq.copy()
                freq_zero['0'] -= 1
                if freq_zero['0'] < 0:
                    continue  # invalid, should not happen
                denom_zero = 1
                for digit in range(10):
                    count = freq_zero.get(str(digit), 0)
                    denom_zero *= math.factorial(count)
                perms_zero_first = math.factorial(total_freq - 1) // denom_zero
            else:
                perms_zero_first = 0
            unique_perms = total_perms - perms_zero_first
            total_count += unique_perms
        
        return total_count