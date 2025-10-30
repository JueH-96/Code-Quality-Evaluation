class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s = list(s)
        n = len(s)
        for i in range(n // 2):
            if s[i] != s[n - i - 1]:
                s[i] = s[n - i - 1] = min(s[i], s[n - i - 1])
        return ''.join(s)