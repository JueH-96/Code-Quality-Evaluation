class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return "9" if k > 1 else "0"
        
        upper = 10**n - 1
        lower = 10**(n-1)
        
        for i in range(upper, lower - 1, -1):
            palindrome = int(str(i) + str(i)[::-1])
            if palindrome % k == 0:
                return str(palindrome)