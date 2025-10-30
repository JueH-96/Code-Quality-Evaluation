class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, 0, -1):
                if i % k == 0:
                    return str(i)

        # Generate the largest palindrome with n digits
        half = (n + 1) // 2
        largest_half = '9' * half
        largest_palindrome = int(largest_half + largest_half[-2::-1])

        # Check if the largest palindrome is divisible by k
        if largest_palindrome % k == 0:
            return str(largest_palindrome)

        # Decrement the largest half and check for the largest k-palindromic number
        largest_half = int(largest_half)
        while largest_half > 0:
            largest_half -= 1
            half_str = str(largest_half).zfill(half)
            candidate = int(half_str + half_str[-2::-1])
            if candidate % k == 0:
                return str(candidate)

        return ""