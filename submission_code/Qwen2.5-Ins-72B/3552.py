class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            return str(k * (k // k))
        
        def is_palindrome(x):
            return x == x[::-1]
        
        def generate_palindrome(n, start):
            half = str(start)
            if n % 2 == 0:
                return half + half[::-1]
            else:
                return half + half[-2::-1]
        
        for i in range(10**((n+1)//2) - 1, 10**((n-1)//2) - 1, -1):
            palindrome = generate_palindrome(n, i)
            if int(palindrome) % k == 0:
                return palindrome
        
        return ""