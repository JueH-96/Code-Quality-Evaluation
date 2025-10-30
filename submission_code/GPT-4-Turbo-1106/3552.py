class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # If n is 1, we can simply return the largest digit less than 10 that is divisible by k
        if n == 1:
            return str((10 - 1) // k * k)
        
        # For n > 1, we construct the largest palindrome with n digits
        # The first half of the palindrome
        first_half = 10**(n - 1) - 1
        
        while first_half > 0:
            # Construct the full palindrome by mirroring the first half
            palindrome_str = str(first_half) + str(first_half)[::-1]
            palindrome = int(palindrome_str)
            
            # Check if the palindrome is divisible by k
            if palindrome % k == 0:
                return palindrome_str
            
            # Decrement the first half to find the next smaller palindrome
            first_half -= 1
        
        # If no palindrome is found, return an empty string
        return ""