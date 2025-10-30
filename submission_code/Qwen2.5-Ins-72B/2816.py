class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s = list(s)
        
        for i in range(n // 2):
            if s[i] != s[n - 1 - i]:
                s[i] = s[n - 1 - i] = min(s[i], s[n - 1 - i])
        
        return ''.join(s)