class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Start from the largest possible number with n digits
        for i in range(10**n - 1, 10**(n-1) - 1, -1):
            # Convert the number to string
            num_str = str(i)
            # Check if the number is a palindrome
            if num_str == num_str[::-1]:
                # Check if the number is divisible by k
                if i % k == 0:
                    return num_str