class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s_list = list(s)
        for i in range(n // 2):
            if s_list[i] != s_list[n - 1 - i]:
                smaller_char = min(s_list[i], s_list[n - 1 - i])
                s_list[i] = smaller_char
                s_list[n - 1 - i] = smaller_char
        return "".join(s_list)