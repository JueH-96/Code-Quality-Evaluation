class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        for i in range(n // 2):
            j = n - 1 - i
            if s_list[i] == s_list[j]:
                continue
            if s_list[i] > s_list[j]:
                s_list[i] = s_list[j]
            else:
                s_list[j] = s_list[i]
        return ''.join(s_list)