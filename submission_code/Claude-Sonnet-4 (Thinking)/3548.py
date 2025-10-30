from math import factorial
from itertools import permutations

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def can_form_palindrome(freq):
            odd_count = sum(1 for f in freq if f % 2 == 1)
            return odd_count <= 1
        
        def has_k_palindromic(freq):
            if not can_form_palindrome(freq):
                return False
            
            # Find the middle digit for odd-length palindromes
            middle_digit = None
            for digit in range(10):
                if freq[digit] % 2 == 1:
                    middle_digit = str(digit)
                    break
            
            # Construct the first half using half the frequency of each digit
            first_half_digits = []
            for digit in range(10):
                first_half_digits.extend([str(digit)] * (freq[digit] // 2))
            
            # Try all unique permutations of the first half
            unique_perms = set(permutations(first_half_digits))
            for perm in unique_perms:
                first_half = ''.join(perm)
                
                # Build the palindrome
                if n % 2 == 0:
                    palindrome = first_half + first_half[::-1]
                else:
                    palindrome = first_half + middle_digit + first_half[::-1]
                
                # Check for no leading zeros and divisibility by k
                if palindrome[0] != '0' and int(palindrome) % k == 0:
                    return True
            
            return False
        
        def count_valid_permutations(freq):
            # Calculate total permutations
            total = factorial(n)
            for f in freq:
                total //= factorial(f)
            
            # Subtract permutations that start with '0'
            if freq[0] > 0:
                temp_freq = freq[:]
                temp_freq[0] -= 1
                starts_with_zero = factorial(n - 1)
                for f in temp_freq:
                    starts_with_zero //= factorial(f)
                total -= starts_with_zero
            
            return total
        
        # Generate all frequency distributions for n digits
        def generate_freq_distributions():
            distributions = []
            
            def recurse(digit, current_freq, remaining):
                if digit == 10:
                    if remaining == 0:
                        distributions.append(current_freq[:])
                    return
                
                for count in range(remaining + 1):
                    current_freq[digit] = count
                    recurse(digit + 1, current_freq, remaining - count)
            
            recurse(0, [0] * 10, n)
            return distributions
        
        total_good_integers = 0
        freq_distributions = generate_freq_distributions()
        
        for freq in freq_distributions:
            if has_k_palindromic(freq):
                valid_permutations = count_valid_permutations(freq)
                total_good_integers += valid_permutations
        
        return total_good_integers