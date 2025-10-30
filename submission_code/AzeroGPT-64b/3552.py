class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, k-1, -1):
                if i % k == 0:
                    return str(i)
        
        start, end = 10**(n-1), 10**n
        for i in range(end-1, start-1, -1):
            num = str(i)
            palindrome = num + num[:-1][::-1]
            if int(palindrome) % k == 0:
                return palindrome
        
        return ""