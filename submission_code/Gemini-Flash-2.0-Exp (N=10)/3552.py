class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for i in range(9, 0, -1):
                if i % k == 0:
                    return str(i)
            return ""

        half_len = (n + 1) // 2
        
        for i in range(10 ** half_len - 1, 10 ** (half_len - 1) - 1, -1):
            s = str(i)
            if n % 2 == 0:
                palindrome = s + s[::-1]
            else:
                palindrome = s + s[:-1][::-1]
            
            if int(palindrome) % k == 0:
                return palindrome
        
        return ""