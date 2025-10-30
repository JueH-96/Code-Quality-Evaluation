class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(9 if k == 1 else 8)
        
        half_length = (n + 1) // 2
        max_half = 10**half_length - 1
        min_half = 10**(half_length - 1)
        
        for half in range(max_half, min_half - 1, -1):
            palindrome = int(str(half) + str(half)[::-1])
            if palindrome % k == 0:
                return str(palindrome)