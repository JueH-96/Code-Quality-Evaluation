class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        from itertools import combinations_with_replacement, permutations
        from math import factorial
        
        result = 0
        seen_configs = set()
        
        # Try all possible digit multisets of size n
        for digit_multiset in combinations_with_replacement(range(10), n):
            digit_count = Counter(digit_multiset)
            
            # Use canonical form to avoid duplicate configurations
            config = tuple(sorted(digit_count.items()))
            if config in seen_configs:
                continue
            seen_configs.add(config)
            
            # Check if this configuration can form a palindrome
            odd_counts = sum(1 for count in digit_count.values() if count % 2 == 1)
            can_be_palindrome = odd_counts <= (n % 2)
            
            if not can_be_palindrome:
                continue
            
            # Check if any palindrome from these digits is divisible by k
            valid_palindrome_exists = False
            
            # Build the components for palindrome construction
            half_digits = []
            middle = None
            for digit, count in digit_count.items():
                half_digits.extend([digit] * (count // 2))
                if count % 2 == 1:
                    middle = digit
            
            # Try all unique permutations of the first half
            for half_perm in set(permutations(half_digits)):
                # Construct the full palindrome
                if n % 2 == 0:
                    full_palindrome = list(half_perm) + list(half_perm[::-1])
                else:
                    full_palindrome = list(half_perm) + [middle] + list(half_perm[::-1])
                
                # Convert to integer and check divisibility
                palindrome_value = 0
                for digit in full_palindrome:
                    palindrome_value = palindrome_value * 10 + digit
                
                if palindrome_value % k == 0:
                    valid_palindrome_exists = True
                    break
            
            if not valid_palindrome_exists:
                continue
            
            # Count how many n-digit numbers have this digit distribution
            total_arrangements = factorial(n)
            for count in digit_count.values():
                total_arrangements //= factorial(count)
            
            # Exclude arrangements with leading zeros
            if 0 in digit_count and digit_count[0] > 0:
                arrangements_with_leading_zero = factorial(n - 1)
                for digit, count in digit_count.items():
                    if digit == 0:
                        arrangements_with_leading_zero //= factorial(count - 1)
                    else:
                        arrangements_with_leading_zero //= factorial(count)
                total_arrangements -= arrangements_with_leading_zero
            
            result += total_arrangements
        
        return result