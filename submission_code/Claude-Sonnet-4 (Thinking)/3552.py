class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def construct_palindrome(first_half_digits, n):
            if n % 2 == 0:
                return ''.join(first_half_digits) + ''.join(first_half_digits[::-1])
            else:
                return ''.join(first_half_digits) + ''.join(first_half_digits[-2::-1])
        
        def string_mod(s, k):
            result = 0
            for char in s:
                result = (result * 10 + int(char)) % k
            return result
        
        # Special cases
        if k == 1:
            return '9' * n
        
        if k == 2:
            if n == 1:
                return '8'
            else:
                return '8' + '9' * (n - 2) + '8'
        
        if k == 3:
            return '9' * n
        
        if k == 5:
            if n == 1:
                return '5'
            else:
                return '5' + '9' * (n - 2) + '5'
        
        # General case using backtracking
        if n % 2 == 0:
            half_len = n // 2
        else:
            half_len = (n + 1) // 2
        
        def backtrack(first_half_digits):
            if len(first_half_digits) == half_len:
                palindrome = construct_palindrome(first_half_digits, n)
                if string_mod(palindrome, k) == 0:
                    return palindrome
                else:
                    return None
            
            start_digit = 1 if len(first_half_digits) == 0 else 0
            for digit in range(9, start_digit - 1, -1):
                first_half_digits.append(str(digit))
                result = backtrack(first_half_digits)
                if result is not None:
                    return result
                first_half_digits.pop()
            
            return None
        
        return backtrack([])