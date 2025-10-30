class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        return ''.join(min(s[i], s[n-1-i]) for i in range(n))