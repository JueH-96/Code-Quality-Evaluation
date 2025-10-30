class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        for i in range(n // 2):
            j = n - 1 - i
            if s_list[i] != s_list[j]:
                min_char = min(s_list[i], s_list[j])
                s_list[i] = min_char
                s_list[j] = min_char
        return ''.join(s_list)