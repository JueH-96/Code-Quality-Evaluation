class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            # For single digit, the largest possible value is 9, we check downwards
            for x in range(9, 0, -1):
                if x % k == 0:
                    return str(x)
        
        # For n > 1, we need to generate the largest n-digit numbers and check if they are k-palindromic
        # Start from the largest n-digit number
        upper_limit = 10**n - 1
        lower_limit = 10**(n-1)
        
        # Generate palindromes from the upper limit
        for first_half in range(upper_limit // 10**(n//2), lower_limit // 10**(n//2) - 1, -1):
            # Create the palindrome by mirroring the first half
            if n % 2 == 0:
                pal_str = str(first_half) + str(first_half)[::-1]
            else:
                pal_str = str(first_half) + str(first_half)[:-1][::-1]
            
            pal_num = int(pal_str)
            
            # Check if this palindrome is divisible by k
            if pal_num % k == 0:
                return pal_str
        
        return ""  # If no such number exists