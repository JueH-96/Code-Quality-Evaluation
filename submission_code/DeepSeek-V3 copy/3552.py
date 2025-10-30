class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            # The largest single-digit number divisible by k is 9 if 9 is divisible by k, else the largest possible.
            for x in range(9, 0, -1):
                if x % k == 0:
                    return str(x)
            return ""
        
        # Generate the largest n-digit palindrome
        # For even n: the first half is mirrored
        # For odd n: the first half is mirrored except the middle character
        half = (n + 1) // 2
        start = 10 ** (half - 1)
        end = 10 ** half
        
        for num in range(end - 1, start - 1, -1):
            s = str(num)
            if n % 2 == 0:
                palindrome = s + s[::-1]
            else:
                palindrome = s + s[:-1][::-1]
            x = int(palindrome)
            if x % k == 0:
                return palindrome
        return ""