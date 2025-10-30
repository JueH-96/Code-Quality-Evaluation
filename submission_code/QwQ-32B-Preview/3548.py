import math
from collections import defaultdict
from itertools import product

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        m = (n + 1) // 2  # digits to fix for palindrome
        factorial = [1] * (n + 1)
        for i in range(1, n + 1):
            factorial[i] = factorial[i - 1] * i
        
        def count_permutations(freq):
            total = factorial[n]
            for f in freq.values():
                total //= factorial[f]
            # Permutations with zero as the first digit
            if freq[0] > 0:
                leading_zero = factorial[n - 1]
                for digit in freq:
                    if digit == 0:
                        leading_zero //= factorial[freq[digit] - 1]
                    else:
                        leading_zero //= factorial[freq[digit]]
                total -= leading_zero
            return total
        
        count = 0
        # Generate all possible first m digits
        for first_half in product(range(1, 10), repeat=1) * product(range(0, 10), repeat=m - 1):
            first_half_digits = list(first_half)
            # Mirror to form the palindrome
            if n % 2 == 0:
                palindrome_str = ''.join(map(str, first_half_digits + first_half_digits[::-1]))
            else:
                palindrome_str = ''.join(map(str, first_half_digits + first_half_digits[-2::-1]))
            palindrome = int(palindrome_str)
            # Check if palindrome is divisible by k
            if palindrome % k == 0:
                # Count the frequency of each digit
                freq = defaultdict(int)
                for char in palindrome_str:
                    freq[int(char)] += 1
                # Count valid permutations without leading zeros
                count += count_permutations(freq)
        
        return count