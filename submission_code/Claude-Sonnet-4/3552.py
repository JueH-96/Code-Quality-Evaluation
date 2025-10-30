class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def construct_palindrome(half):
            if n % 2 == 1:
                return half + half[-2::-1]
            else:
                return half + half[::-1]
        
        def is_divisible(half):
            palindrome = construct_palindrome(half)
            return int(palindrome) % k == 0
        
        # Start with the largest possible half
        half_len = (n + 1) // 2
        half = [9] * half_len
        
        # Convert to integer for easier manipulation
        half_num = int(''.join(map(str, half)))
        
        # If n == 1, handle special case
        if n == 1:
            for digit in range(9, -1, -1):
                if digit % k == 0:
                    return str(digit)
        
        # For multi-digit numbers, ensure no leading zeros
        min_half = 10 ** (half_len - 1)  # Minimum value to avoid leading zeros
        
        # Try decreasing half_num until we find a divisible palindrome
        while half_num >= min_half:
            half_str = str(half_num).zfill(half_len)
            palindrome = construct_palindrome(half_str)
            
            if int(palindrome) % k == 0:
                return palindrome
            
            half_num -= 1
        
        # This shouldn't happen given the constraints
        return ""