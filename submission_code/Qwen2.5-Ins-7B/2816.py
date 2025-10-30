class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        result = list(s)
        for i in range(n // 2):
            if s[i] != s[n - i - 1]:
                result[i] = result[n - i - 1] = min(s[i], s[n - i - 1])
        return ''.join(result)