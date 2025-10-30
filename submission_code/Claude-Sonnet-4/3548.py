class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        from math import factorial
        
        def get_palindrome_from_half(half_digits, n):
            # Create palindrome from first half
            if n % 2 == 0:
                return half_digits + half_digits[::-1]
            else:
                return half_digits + half_digits[-2::-1]
        
        def count_arrangements(digit_count, total_digits):
            # Count arrangements avoiding leading zeros
            total_arrangements = factorial(total_digits)
            for count in digit_count.values():
                total_arrangements //= factorial(count)
            
            # Subtract arrangements with leading zero
            if digit_count[0] > 0:
                digit_count[0] -= 1
                arrangements_with_leading_zero = factorial(total_digits - 1)
                for count in digit_count.values():
                    arrangements_with_leading_zero //= factorial(count)
                digit_count[0] += 1
                return total_arrangements - arrangements_with_leading_zero
            
            return total_arrangements
        
        seen_patterns = set()
        result = 0
        
        # Generate all possible first halves
        half_length = (n + 1) // 2
        
        # Start from appropriate range to avoid leading zeros
        start = 10 ** (half_length - 1) if half_length > 1 else 1
        end = 10 ** half_length
        
        for half in range(start, end):
            half_str = str(half).zfill(half_length)
            palindrome_str = get_palindrome_from_half(half_str, n)
            palindrome_num = int(palindrome_str)
            
            # Check if divisible by k
            if palindrome_num % k != 0:
                continue
            
            # Create canonical representation (sorted digit string)
            canonical = ''.join(sorted(palindrome_str))
            
            if canonical in seen_patterns:
                continue
            
            seen_patterns.add(canonical)
            
            # Count digits
            digit_count = [0] * 10
            for digit in palindrome_str:
                digit_count[int(digit)] += 1
            
            # Count valid arrangements
            arrangements = count_arrangements(digit_count, n)
            result += arrangements
        
        return result