class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, k-1, -1):
                if i % k == 0:
                    return str(i)
            return str(k)
        
        upper_limit = 10**n - 1
        lower_limit = 10**(n-1)
        
        for i in range(upper_limit, lower_limit-1, -1):
            half = str(i)[:len(str(i))//2+1]
            palindrome = int(half + half[-2::-1])
            if palindrome % k == 0:
                return str(palindrome)
        
        return str(k)