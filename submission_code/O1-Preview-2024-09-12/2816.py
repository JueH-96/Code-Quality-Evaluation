class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        result = list(s)
        n = len(s)
        for i in range(n // 2):
            if result[i] != result[n - 1 - i]:
                min_char = min(result[i], result[n - 1 - i])
                result[i] = result[n - 1 - i] = min_char
        return ''.join(result)