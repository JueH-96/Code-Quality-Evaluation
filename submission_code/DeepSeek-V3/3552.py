class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            # For single-digit numbers, the largest divisible by k is the largest single-digit number divisible by k
            for x in range(9, -1, -1):
                if x % k == 0:
                    return str(x)
        # Generate the largest possible n-digit palindrome
        # For even n: the first half is mirrored
        # For odd n: the first half is mirrored except the middle character
        half_len = (n + 1) // 2
        start = 10 ** (half_len - 1)
        end = 10 ** half_len
        for first_half in range(end - 1, start - 1, -1):
            s = str(first_half)
            if n % 2 == 0:
                palindrome = s + s[::-1]
            else:
                palindrome = s + s[:-1][::-1]
            x = int(palindrome)
            if x % k == 0:
                return palindrome
        return ""