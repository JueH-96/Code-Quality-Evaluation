class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from math import factorial
        from collections import Counter
        
        def can_form_palindrome(freq):
            odd_count = sum(1 for count in freq.values() if count % 2 == 1)
            return odd_count <= 1
        
        def count_permutations(freq, n):
            # Count permutations without leading zeros
            total = factorial(n)
            for count in freq.values():
                total //= factorial(count)
            
            # Subtract permutations with leading zeros
            if freq.get(0, 0) > 0:
                freq_copy = freq.copy()
                freq_copy[0] -= 1
                zeros_first = factorial(n - 1)
                for digit, count in freq_copy.items():
                    zeros_first //= factorial(count)
                total -= zeros_first
            
            return total
        
        def generate_palindrome_from_freq(freq, n):
            # Generate a palindrome from frequency map
            half = []
            middle = -1
            
            for digit, count in sorted(freq.items()):
                if count % 2 == 1:
                    middle = digit
                    half.extend([digit] * (count // 2))
                else:
                    half.extend([digit] * (count // 2))
            
            if n % 2 == 1:
                palindrome = half + [middle] + half[::-1]
            else:
                palindrome = half + half[::-1]
            
            # Convert to number
            num = 0
            for d in palindrome:
                num = num * 10 + d
            return num
        
        # Generate all possible digit frequencies
        def generate_frequencies(n, current_freq, digit, remaining):
            if digit == 10:
                if remaining == 0:
                    yield current_freq.copy()
                return
            
            for count in range(remaining + 1):
                if count > 0:
                    current_freq[digit] = count
                yield from generate_frequencies(n, current_freq, digit + 1, remaining - count)
                if count > 0:
                    del current_freq[digit]
        
        valid_patterns = set()
        
        for freq in generate_frequencies(n, {}, 0, n):
            if not can_form_palindrome(freq):
                continue
            
            # Check if we can form a valid palindrome divisible by k
            # Try different arrangements to avoid leading zeros in palindrome
            if n == 1:
                # Single digit case
                for digit in freq:
                    if digit % k == 0:
                        valid_patterns.add(tuple(sorted(freq.items())))
                        break
            else:
                # Multi-digit case - need to check if palindrome is divisible by k
                # and doesn't have leading zeros
                digits = []
                for digit, count in freq.items():
                    digits.extend([digit] * count)
                
                # Try to build a palindrome
                half_digits = []
                middle_digit = -1
                digit_counts = Counter(digits)
                
                for digit, count in digit_counts.items():
                    if count % 2 == 1:
                        middle_digit = digit
                    half_digits.extend([digit] * (count // 2))
                
                # Check all permutations of half_digits to find valid palindrome
                from itertools import permutations
                half_perms = set(permutations(half_digits))
                
                for half_perm in half_perms:
                    if len(half_perm) > 0 and half_perm[0] == 0:
                        continue
                    
                    if n % 2 == 1:
                        palindrome_digits = list(half_perm) + [middle_digit] + list(half_perm[::-1])
                    else:
                        palindrome_digits = list(half_perm) + list(half_perm[::-1])
                    
                    # Convert to number
                    num = 0
                    for d in palindrome_digits:
                        num = num * 10 + d
                    
                    if num % k == 0:
                        valid_patterns.add(tuple(sorted(freq.items())))
                        break
        
        # Count total good integers
        total = 0
        for pattern in valid_patterns:
            freq = dict(pattern)
            total += count_permutations(freq, n)
        
        return total