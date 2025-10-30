class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Special case for single digit numbers
        if n == 1:
            for digit in range(9, 0, -1):
                if digit % k == 0:
                    return str(digit)
        
        # For n > 1
        if n % 2 == 0:  # Even number of digits
            half_digits = n // 2
            max_left_half = 10**half_digits - 1
            min_left_half = 10**(half_digits - 1)
            
            for left_half in range(max_left_half, min_left_half - 1, -1):
                palindrome = str(left_half) + str(left_half)[::-1]
                if int(palindrome) % k == 0:
                    return palindrome
        else:  # Odd number of digits
            half_digits = n // 2
            max_left_half = 10**half_digits - 1
            min_left_half = 10**(half_digits - 1)
            
            for left_half in range(max_left_half, min_left_half - 1, -1):
                for middle_digit in range(9, -1, -1):
                    palindrome = str(left_half) + str(middle_digit) + str(left_half)[::-1]
                    if int(palindrome) % k == 0:
                        return palindrome
        
        return ""  # Should not happen with given constraints