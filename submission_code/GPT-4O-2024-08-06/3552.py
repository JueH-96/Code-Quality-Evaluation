class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Start with the largest n-digit number
        max_num = 10**n - 1
        
        # Iterate downwards to find the largest k-palindromic number
        for num in range(max_num, 10**(n-1) - 1, -1):
            # Convert the number to a string and check if it's a palindrome
            if str(num) == str(num)[::-1]:
                # Check if the number is divisible by k
                if num % k == 0:
                    return str(num)
        
        # If no k-palindromic number is found, return an empty string
        return ""