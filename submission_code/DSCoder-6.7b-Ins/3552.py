class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(k%10)
        if k%2 == 0:
            first_half = int('9'*n)
        else:
            first_half = int('9'*(n-1) + '8')
        while True:
            palindrome = int(str(first_half) + str(first_half)[::-1])
            if palindrome%k == 0:
                return str(palindrome)
            first_half -= 1