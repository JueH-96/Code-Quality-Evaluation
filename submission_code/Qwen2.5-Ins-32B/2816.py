class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s = list(s)
        for i in range(n // 2):
            if s[i] != s[n - i - 1]:
                if s[i] < s[n - i - 1]:
                    s[n - i - 1] = s[i]
                else:
                    s[i] = s[n - i - 1]
        return ''.join(s)