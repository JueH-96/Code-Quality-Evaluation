class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Generate the largest possible palindrome with n digits
        upper_bound = 10 ** n - 1
        lower_bound = 10 ** (n - 1)
        
        for num in range(upper_bound, lower_bound - 1, -1):
            # Check if the number is a palindrome
            palindrome = int(str(num) + str(num)[::-1])
            
            # Check if the palindrome is divisible by k
            if palindrome % k == 0:
                return str(palindrome)
        
        # If no k-palindromic number is found, return the largest possible k-palindromic number
        return str(10 ** (n - 1) * k)