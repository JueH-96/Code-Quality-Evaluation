class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s = list(s)
        for i, j in enumerate(s[:len(s)//2]):
            s[-(i+1)] = min(j, s[-(i+1)])
        return ''.join(s)